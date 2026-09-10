"""Read-only fixed-scope Athena collection; exclusive local evidence outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-socket", required=True)
    parser.add_argument("--local-parent", required=True)
    args = parser.parse_args()
    parent = Path(args.local_parent).resolve(strict=True)
    socket = Path(args.control_socket)
    if not socket.is_absolute() or not socket.exists():
        raise ValueError("An existing absolute shared SSH control socket is required")
    script = Path(__file__).with_name("baseline_export.py").read_bytes()
    output = Path(tempfile.mkdtemp(prefix="athena_u_raw_", dir=parent))
    bundle = output / "bundle.json"
    error_log = output / "collector.stderr"
    print(json.dumps({"local_output": str(output), "remote_writes": False}), flush=True)
    command = ["ssh", "-S", str(socket), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               "jp673@athena.egr.duke.edu", "python3 -B - collect"]
    transport_error = None
    returncode = None
    with bundle.open("xb") as stdout, error_log.open("xb") as stderr:
        try:
            result = subprocess.run(command, input=script, stdout=stdout, stderr=stderr, timeout=600)
            returncode = result.returncode
        except (subprocess.TimeoutExpired, OSError) as error:
            transport_error = {"type": type(error).__name__, "message": str(error)}
    receipt = {
        "collector_sha256": hashlib.sha256(script).hexdigest(),
        "command": command, "returncode": returncode,
        "transport_error": transport_error,
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "bundle_bytes": bundle.stat().st_size,
        "stderr_sha256": hashlib.sha256(error_log.read_bytes()).hexdigest(),
        "remote_writes": False,
    }
    with (output / "transport_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if returncode != 0 or transport_error is not None:
        print(error_log.read_text(errors="replace")[-12000:], flush=True)
        raise SystemExit(returncode if returncode is not None else 1)


if __name__ == "__main__":
    main()
