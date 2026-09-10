"""Fixed-scope read-only Athena inventory; no backend import or job submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile

ROOT = Path("/zpool-00/home/jp673/robomme_repro/robomme_policy_learning")
BENCHMARK = ROOT / "third_party/robomme_benchmark"
CHECKPOINT = ROOT / "runs/test_time_scaling/checkpoints/perceptual-framesamp-modul/79999"


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=45,
                                env=dict(os.environ, GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1"))
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": type(error).__name__, "message": str(error)}


def small_file(path):
    if not path.is_file():
        return {"path": str(path), "exists": False}
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Inventory must not read large artifacts")
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def collect():
    if socket.gethostname() not in {"athena", "athena.egr.duke.edu"}:
        raise ValueError("This read-only collector is limited to the named Athena login host")
    result = {"kind": "read_only_runtime_inventory_not_launch_evidence", "schema_version": 1,
              "observed_utc": datetime.now(timezone.utc).isoformat(),
              "host": socket.gethostname(), "uid": os.getuid(),
              "remote_writes": False, "gpu_execution": False, "repositories": {}, "runtimes": {}}
    for role, root in (("policy", ROOT), ("benchmark", BENCHMARK)):
        item = {"path": str(root), "exists": root.is_dir(), "resolved_path": str(root.resolve()),
                "lockfile": small_file(root / "uv.lock")}
        if root.is_dir():
            item["revision"] = command(["git", "-C", str(root), "rev-parse", "HEAD"])
            item["branch"] = command(["git", "-C", str(root), "branch", "--show-current"])
            item["status"] = command(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=normal"])
        result["repositories"][role] = item
        executable = root / ".venv/bin/python"
        code = (
            "import importlib.metadata as m,json,sys,platform; "
            "names=['numpy','jax','jaxlib','torch','flax','omegaconf','mani-skill','sapien']; "
            "available={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}; "
            "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'python_version':platform.python_version(),"
            "'packages':{name:available.get(name) for name in names}}))"
        )
        result["runtimes"][role] = {"python": command([str(executable), "-B", "-c", code]),
                                    "venv_config": small_file(root / ".venv/pyvenv.cfg")}
    history = small_file(CHECKPOINT.parent / "history_config.txt")
    if "sha256" in history:
        history["text"] = (CHECKPOINT.parent / "history_config.txt").read_text()
    archive = CHECKPOINT.with_suffix(".zip")
    result["checkpoint"] = {"path": str(CHECKPOINT), "exists": CHECKPOINT.is_dir(),
                            "history_config": history, "params_exists": (CHECKPOINT / "params").is_dir(),
                            "archive_path": str(archive), "archive_exists": archive.is_file(),
                            "archive_size_bytes": archive.stat().st_size if archive.is_file() else None,
                            "parameter_bytes_hashed": False}
    result["own_jobs"] = command(["squeue", "--noheader", "--user", "jp673",
                                   "--format=%i|%j|%T|%M|%D|%R"])
    result["partition_inventory"] = command(["sinfo", "--noheader", "--format=%P|%a|%l|%D|%G"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-socket")
    parser.add_argument("--local-parent")
    args = parser.parse_args()
    if args.control_socket is None and args.local_parent is None:
        print(json.dumps(collect(), sort_keys=True, indent=2))
        return
    if args.control_socket is None or args.local_parent is None:
        parser.error("Local collection requires both control socket and destination parent")
    parent = Path(args.local_parent).resolve(strict=True)
    control = Path(args.control_socket)
    if not control.is_absolute() or not control.exists():
        raise ValueError("An existing absolute shared SSH socket is required")
    directory = Path(tempfile.mkdtemp(prefix="athena_runtime_", dir=parent))
    script = Path(__file__).read_bytes()
    argv = ["ssh", "-S", str(control), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "jp673@athena.egr.duke.edu", "python3 -B -"]
    status, error = None, None
    with (directory / "inventory.json").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        try:
            status = subprocess.run(argv, input=script, stdout=stdout, stderr=stderr, timeout=300).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    receipt = {"collector_sha256": hashlib.sha256(script).hexdigest(), "returncode": status,
               "error": error, "output": str(directory), "remote_writes": False,
               "inventory_sha256": hashlib.sha256((directory / "inventory.json").read_bytes()).hexdigest()}
    with (directory / "transport_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if status != 0:
        print((directory / "stderr.txt").read_text(errors="replace")[-8000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
