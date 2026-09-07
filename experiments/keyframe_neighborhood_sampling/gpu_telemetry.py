"""Optional sampled NVIDIA telemetry; stdlib only and never a scientific metric.

The monitor queries only explicitly selected physical GPU UUIDs. It performs no
allocation, process-owner query, device mutation, or resource substitution.
Peaks are maxima of successful samples, not exact peaks between samples.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


class GpuTelemetry:
    """Append observations to a new exclusive JSONL while this context is active.

    Construction is inert; entering creates the log and starts one CPU thread.
    Existing logs are never reused. Query/parse and subsequent logging failures
    are observational only: they never raise into the experiment body or its
    cleanup. Inspect ``summary`` after exit for sampled peaks and logger status.
    """

    def __init__(
        self,
        path: Path,
        gpu_uuids: tuple[str, ...] | list[str],
        *,
        interval_seconds: float = 2.0,
        query_timeout_seconds: float = 2.0,
    ) -> None:
        if (
            not gpu_uuids
            or isinstance(gpu_uuids, str)
            or any(not isinstance(gpu, str) or GPU_UUID.fullmatch(gpu) is None for gpu in gpu_uuids)
            or len(set(gpu_uuids)) != len(gpu_uuids)
        ):
            raise ValueError("Telemetry requires unique full physical GPU UUIDs")
        if isinstance(interval_seconds, bool) or not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Telemetry interval must be finite and positive")
        if (
            isinstance(query_timeout_seconds, bool)
            or not math.isfinite(query_timeout_seconds)
            or not 0 < query_timeout_seconds <= 10
        ):
            raise ValueError("Telemetry query timeout must be positive and at most 10 seconds")
        self.path = Path(path)
        self.gpu_uuids = tuple(gpu_uuids)
        self.interval_seconds = float(interval_seconds)
        self.query_timeout_seconds = float(query_timeout_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._state = "new"
        self._started_monotonic: float | None = None
        self._samples = 0
        self._query_errors = 0
        self._logging_error: str | None = None
        self._thread_stopped: bool | None = None
        self._peaks = {
            gpu: {"sample_count": 0, "sampled_peak_memory_used_mib": None, "sampled_peak_utilization_gpu_percent": None}
            for gpu in self.gpu_uuids
        }

    def _record(self, kind: str, **fields) -> dict:
        return {
            "kind": kind,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
            "elapsed_seconds": (
                None if self._started_monotonic is None else max(0.0, time.monotonic() - self._started_monotonic)
            ),
            **fields,
        }

    def _write_locked(self, record: dict) -> None:
        if self._stream is None:
            return
        try:
            self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            self._stream.flush()
        except Exception as error:
            # Never log exception text or command output: it could include a
            # device outside the selected set, and cannot repair a failed log.
            self._logging_error = type(error).__name__
            self._stop.set()

    def _summary_locked(self) -> dict:
        return {
            "gpu_uuids": list(self.gpu_uuids),
            "peak_semantics": "maximum_of_successful_samples_not_exact_peak",
            "successful_sample_count": self._samples,
            "query_error_count": self._query_errors,
            "sampled_peaks": {gpu: dict(values) for gpu, values in self._peaks.items()},
            "logging_error": self._logging_error,
            "thread_stopped": self._thread_stopped,
            "state": self._state,
        }

    @property
    def summary(self) -> dict:
        with self._lock:
            return self._summary_locked()

    def _query(self) -> list[dict]:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={','.join(self.gpu_uuids)}",
                "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.query_timeout_seconds,
        )
        samples = {}
        for fields in csv.reader(io.StringIO(result.stdout)):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) != 4:
                raise ValueError("Unexpected GPU telemetry fields")
            gpu, total, used, utilization = (field.strip() for field in fields)
            if gpu not in self._peaks or gpu in samples:
                raise ValueError("Unexpected or duplicate GPU telemetry identity")
            total, used, utilization = int(total), int(used), int(utilization)
            if total <= 0 or not 0 <= used <= total or not 0 <= utilization <= 100:
                raise ValueError("Invalid GPU telemetry measurements")
            samples[gpu] = {
                "gpu_uuid": gpu,
                "memory_total_mib": total,
                "memory_used_mib": used,
                "utilization_gpu_percent": utilization,
            }
        if set(samples) != set(self.gpu_uuids):
            raise ValueError("Missing selected GPU telemetry identity")
        return [samples[gpu] for gpu in self.gpu_uuids]

    def _run(self) -> None:
        while not self._stop.is_set():
            query_started = time.monotonic()
            try:
                samples = self._query()
            except Exception as error:
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._query_errors += 1
                    fields = {"gpu_uuids": list(self.gpu_uuids), "error_type": type(error).__name__}
                    if isinstance(error, subprocess.CalledProcessError):
                        fields["returncode"] = error.returncode
                    self._write_locked(self._record("query_error", **fields))
            else:
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._samples += 1
                    for sample in samples:
                        peak = self._peaks[sample["gpu_uuid"]]
                        peak["sample_count"] += 1
                        for peak_field, sample_field in (
                            ("sampled_peak_memory_used_mib", "memory_used_mib"),
                            ("sampled_peak_utilization_gpu_percent", "utilization_gpu_percent"),
                        ):
                            prior = peak[peak_field]
                            peak[peak_field] = (
                                sample[sample_field] if prior is None else max(prior, sample[sample_field])
                            )
                    self._write_locked(self._record("sample", gpus=samples))
            self._stop.wait(max(0.0, self.interval_seconds - (time.monotonic() - query_started)))

    def start(self) -> GpuTelemetry:
        with self._lock:
            if self._state != "new":
                raise RuntimeError("Telemetry contexts cannot be restarted or reused")
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
            )
            try:
                self._stream = os.fdopen(descriptor, "a", encoding="utf-8")
            except BaseException:
                os.close(descriptor)
                raise
            self._state = "running"
            self._started_monotonic = time.monotonic()
            self._write_locked(
                self._record(
                    "start",
                    gpu_uuids=list(self.gpu_uuids),
                    sampling_interval_seconds=self.interval_seconds,
                    query_timeout_seconds=self.query_timeout_seconds,
                    peak_semantics="maximum_of_successful_samples_not_exact_peak",
                )
            )
            self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._state = "closed"
                self._stream.close()
                self._stream = None
                self._thread = None
                raise
        return self

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            # Normal query timeouts reap the owned nvidia-smi subprocess. Do
            # not block experiment cleanup indefinitely if the OS cannot do so.
            thread.join(timeout=self.query_timeout_seconds + 1.0)
        with self._lock:
            if self._state == "closed":
                return
            self._state = "closed"
            self._thread_stopped = thread is None or not thread.is_alive()
            if not self._thread_stopped:
                self._logging_error = self._logging_error or "TelemetryThreadJoinTimeout"
            self._write_locked(self._record("summary", **self._summary_locked()))
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception as error:
                    self._logging_error = type(error).__name__
                self._stream = None

    def __enter__(self) -> GpuTelemetry:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
