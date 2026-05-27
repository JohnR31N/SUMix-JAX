import jax.numpy as jnp
from flax import linen as nn


class SUMixHead(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, features):
        cls_logits = nn.Dense(
            self.num_classes,
            name="classifier",
        )(features)

        uncertain_logits = nn.Dense(
            self.num_classes,
            name="uncertainty_classifier",
        )(features)

        return cls_logits, uncertain_logits