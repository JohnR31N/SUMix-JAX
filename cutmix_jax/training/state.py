from typing import Any, Dict

import jax.numpy as jnp
from flax.training import train_state

from cutmix_jax.training.config import create_optimizer


class TrainState(train_state.TrainState):
    batch_stats: Dict[str, Any]


def create_train_state(rng, model, args, steps_per_epoch: int):
    dummy_x = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    variables = model.init(rng, dummy_x, train=True)
    tx = create_optimizer(args=args, steps_per_epoch=steps_per_epoch)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )


def get_logits(outputs):
    """
    Normal model returns:
        logits

    SUMix model returns:
        cls_logits, uncertain_logits

    This helper extracts cls_logits.
    """
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs
