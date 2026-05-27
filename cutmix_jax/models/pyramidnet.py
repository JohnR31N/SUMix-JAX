from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class PyramidBottleneckBlock(nn.Module):
    planes: int
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        shortcut = x

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.planes,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        shortcut = self._shortcut(shortcut, x)

        return x + shortcut

    def _shortcut(self, shortcut, residual):
        if self.stride != 1:
            shortcut = nn.avg_pool(
                shortcut,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            )

        in_channels = shortcut.shape[-1]
        out_channels = residual.shape[-1]

        if in_channels < out_channels:
            pad_channels = out_channels - in_channels
            padding = jnp.zeros(
                shortcut.shape[:-1] + (pad_channels,),
                dtype=shortcut.dtype,
            )
            shortcut = jnp.concatenate([shortcut, padding], axis=-1)

        elif in_channels > out_channels:
            shortcut = shortcut[..., :out_channels]

        return shortcut


class PyramidBasicBlock(nn.Module):
    out_channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        shortcut = x

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        shortcut = self._shortcut(shortcut, x)

        return x + shortcut

    def _shortcut(self, shortcut, residual):
        if self.stride != 1:
            shortcut = nn.avg_pool(
                shortcut,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="VALID",
            )

        in_channels = shortcut.shape[-1]
        out_channels = residual.shape[-1]

        if in_channels < out_channels:
            pad_channels = out_channels - in_channels
            padding = jnp.zeros(
                shortcut.shape[:-1] + (pad_channels,),
                dtype=shortcut.dtype,
            )
            shortcut = jnp.concatenate([shortcut, padding], axis=-1)

        elif in_channels > out_channels:
            shortcut = shortcut[..., :out_channels]

        return shortcut


class PyramidNet(nn.Module):
    depth: int = 20
    alpha: int = 48
    num_classes: int = 10
    bottleneck: bool = False

    def setup(self):
        if self.bottleneck:
            if (self.depth - 2) % 9 != 0:
                raise ValueError(
                    "For bottleneck PyramidNet, depth should satisfy depth = 9n + 2. "
                    "Example: depth=200."
                )
            self.blocks_per_stage = (self.depth - 2) // 9
            self.expansion = 4
        else:
            if (self.depth - 2) % 6 != 0:
                raise ValueError(
                    "For basic PyramidNet, depth should satisfy depth = 6n + 2. "
                    "Example: depth=20, 32, 44."
                )
            self.blocks_per_stage = (self.depth - 2) // 6
            self.expansion = 1

        self.total_blocks = self.blocks_per_stage * 3
        self.add_rate = self.alpha / self.total_blocks

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)

        feature_dim = 16.0

        x, feature_dim = self._make_stage(
            x=x,
            feature_dim=feature_dim,
            stage_index=0,
            stride=1,
            train=train,
        )

        x, feature_dim = self._make_stage(
            x=x,
            feature_dim=feature_dim,
            stage_index=1,
            stride=2,
            train=train,
        )

        x, feature_dim = self._make_stage(
            x=x,
            feature_dim=feature_dim,
            stage_index=2,
            stride=2,
            train=train,
        )

        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)

        return x

    def _make_stage(
        self,
        x,
        feature_dim: float,
        stage_index: int,
        stride: int,
        train: bool,
    ) -> Tuple[jnp.ndarray, float]:
        for block_index in range(self.blocks_per_stage):
            feature_dim += self.add_rate

            block_stride = stride if block_index == 0 else 1

            if self.bottleneck:
                planes = int(round(feature_dim))
                out_channels = planes * self.expansion

                x = PyramidBottleneckBlock(
                    planes=planes,
                    out_channels=out_channels,
                    stride=block_stride,
                    name=f"stage{stage_index}_block{block_index}",
                )(x, train=train)

            else:
                out_channels = int(round(feature_dim))

                x = PyramidBasicBlock(
                    out_channels=out_channels,
                    stride=block_stride,
                    name=f"stage{stage_index}_block{block_index}",
                )(x, train=train)

        return x, feature_dim