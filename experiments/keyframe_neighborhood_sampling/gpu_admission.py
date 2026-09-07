"""Explicit, recorded admission rules for user-authorized GPU sharing."""

from collections.abc import Mapping


def admission_policy(profile: Mapping) -> dict:
    """Legacy jobs remain exclusive; sharing cannot be inferred from free VRAM."""
    policy = profile.get("gpu_admission")
    if policy is None:
        return {"mode": "exclusive"}
    expected = {"mode", "min_free_memory_mib", "max_utilization_gpu_percent"}
    if not isinstance(policy, Mapping) or set(policy) != expected or policy["mode"] != "shared":
        raise ValueError("Invalid explicit GPU sharing policy")
    if profile.get("gpu_layout") != "colocated":
        raise ValueError("Shared GPU admission requires colocated placement and non-preallocating policy")
    minimum, maximum = policy["min_free_memory_mib"], policy["max_utilization_gpu_percent"]
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("Shared GPU minimum free memory must be a positive integer in MiB")
    if type(maximum) is not int or not 0 <= maximum <= 100:
        raise ValueError("Shared GPU maximum utilization must be an integer in 0..100")
    return dict(policy)
