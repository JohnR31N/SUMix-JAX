from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


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

    @nn.compact
    def __call__(self, x, train: bool = True):
        # CIFAR-style ResNet stem: 3x3 conv, no 7x7 conv, no maxpool.
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

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)

        return x


class ResNet18(ResNet):
    stage_sizes: Sequence[int] = (2, 2, 2, 2)
