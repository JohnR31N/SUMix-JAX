import jax


def metric_to_float(metrics, key: str) -> float:
    """Convert scalar metric from either jit or pmap output to Python float."""
    value = jax.device_get(metrics[key])
    if hasattr(value, "shape") and value.shape != ():
        return float(value[0])
    return float(value)
