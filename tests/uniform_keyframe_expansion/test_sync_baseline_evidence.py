"""Transport-only CPU fixtures: never contact SSH or validate fabricated U data."""
import json
from types import SimpleNamespace

import pytest

from experiments.uniform_keyframe_expansion import sync_baseline_evidence as sync


@pytest.mark.parametrize("outcome", [0, 9, "timeout", "missing"])
def test_fixed_readonly_transport_preserves_success_or_failure(tmp_path, monkeypatch, outcome):
    socket = tmp_path / "socket-fixture"
    socket.touch()
    parent = tmp_path / "outputs"
    parent.mkdir()
    monkeypatch.setattr("sys.argv", ["collect", "--control-socket", str(socket), "--local-parent", str(parent)])
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        assert command[-1] == "python3 -B - collect"
        assert isinstance(kwargs["input"], bytes)
        kwargs["stdout"].write(b"fixture data, not an actual U bundle")
        kwargs["stderr"].write(b"fixture stderr")
        if outcome == "timeout":
            raise sync.subprocess.TimeoutExpired(command, 600)
        if outcome == "missing":
            raise FileNotFoundError("fixture ssh missing")
        return SimpleNamespace(returncode=outcome)
    monkeypatch.setattr(sync.subprocess, "run", run)
    if outcome == 0:
        sync.main()
    else:
        with pytest.raises(SystemExit):
            sync.main()
    [directory] = list(parent.iterdir())
    receipt = json.loads((directory / "transport_receipt.json").read_text())
    assert receipt["remote_writes"] is False
    assert receipt["returncode"] == (outcome if type(outcome) is int else None)
    assert bool(receipt["transport_error"]) is (type(outcome) is str)
    assert (directory / "bundle.json").read_bytes() == b"fixture data, not an actual U bundle"
    # A second attempt reserves a separate directory, preserving the first.
    try:
        sync.main()
    except SystemExit:
        pass
    assert len(list(parent.iterdir())) == 2
    assert len(seen) == 2
