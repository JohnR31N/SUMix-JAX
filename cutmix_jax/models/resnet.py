from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from cutmix_jax.models.sumix_head import SUMixHead


class BasicBlock(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        shortcut = x

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if shortcut.shape != x.shape:
            shortcut = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(shortcut)
            shortcut = nn.BatchNorm(use_running_average=not train)(shortcut)

        x = x + shortcut
        x = nn.relu(x)

        return x


class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    num_classes: int = 10
    base_width: int = 64
    use_sumix_head: bool = False

    @nn.compact
    def __call__(
        self,
        x,
        train: bool = True,
        return_features: bool = False,
    ):
        x = nn.Conv(
            features=self.base_width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        widths = [
            self.base_width,
            self.base_width * 2,
            self.base_width * 4,
            self.base_width * 8,
        ]

        for stage_idx, block_count in enumerate(self.stage_sizes):
            for block_idx in range(block_count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                x = BasicBlock(
                    features=widths[stage_idx],
                    stride=stride,
                    name=f"stage{stage_idx}_block{block_idx}",
                )(x, train=train)

        feature_map = x
        features = jnp.mean(feature_map, axis=(1, 2))

        if self.use_sumix_head:
            cls_logits, uncertain_logits = SUMixHead(
                num_classes=self.num_classes,
                name="sumix_head",
            )(features)

            if return_features:
                return cls_logits, uncertain_logits, features

            return cls_logits, uncertain_logits

        logits = nn.Dense(
            self.num_classes,
            name="classifier",
        )(features)

        if return_features:
            return logits, features

        return logits


def ResNet18(
    num_classes: int = 10,
    base_width: int = 64,
    use_sumix_head: bool = False,
):
    return ResNet(
        stage_sizes=(2, 2, 2, 2),
        num_classes=num_classes,
        base_width=base_width,
        use_sumix_head=use_sumix_head,
    )