import dataclasses
import enum
import logging
from pathlib import Path
import socket
from typing import TYPE_CHECKING

import tyro

from mme_vla_suite.serving import websocket_policy_server
from openpi.serving.websocket_policy_server import adopt_listening_socket
from openpi.serving.websocket_policy_server import validate_direct_options

if TYPE_CHECKING:
    from mme_vla_suite.policies import policy as _policy


class EnvMode(enum.Enum):
    """Supported environments."""

    HISTORY_BENCH = "history_bench"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

    def __post_init__(self):
        self.dir = Path(self.dir)


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.HISTORY_BENCH

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on. Ignored when listen_fd is supplied.
    port: int = 8000
    # Optional inherited, already-listening loopback TCP socket. The caller retains its reservation.
    listen_fd: int | None = None
    # Direct-runner readiness identity. Both fields are required with listen_fd.
    execution_id: str | None = None
    dispatch_sha256: str | None = None
    # Record the policy's behavior for debugging.
    record: bool = False
    seed: int = 42

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    def __post_init__(self) -> None:
        validate_direct_options(self.listen_fd, self.execution_id, self.dispatch_sha256)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.HISTORY_BENCH: Checkpoint(
        config="mme_vla_suite",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    )
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> "_policy.MME_VLA_Policy":
    """Create a default policy for the given environment."""
    from mme_vla_suite.policies import policy_config as _policy_config  # noqa: PLC0415 - validate transport first
    from mme_vla_suite.training import config as _config  # noqa: PLC0415

    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> "_policy.MME_VLA_Policy":
    """Create a policy from the given arguments."""
    from mme_vla_suite.policies import policy_config as _policy_config  # noqa: PLC0415 - validate transport first
    from mme_vla_suite.training import config as _config  # noqa: PLC0415

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                seed=args.seed,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    validate_direct_options(args.listen_fd, args.execution_id, args.dispatch_sha256)
    if args.listen_fd is not None:
        # Fail invalid descriptors before importing/loading model dependencies.
        # This preflight closes only a duplicate; the reservation stays held.
        with adopt_listening_socket(args.listen_fd):
            pass
    policy = create_policy(args)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        listen_fd=args.listen_fd,
        execution_id=args.execution_id,
        dispatch_sha256=args.dispatch_sha256,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
