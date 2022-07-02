from typing import Any, List, Type, Union

import torch

from torch import Tensor
from torch.nn import (
    Module,
    AdaptiveAvgPool1d,
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    GELU,
    Identity,
    LayerNorm,
    Linear,
    MultiheadAttention,
    Sequential,
)


class PoolMixer(Module):
    def __init__(self, kernel_size: int = 3):
        super(PoolMixer, self).__init__()
        self.pool = AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.pool(x)


class MetaBlock4D(Module):
    # "EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>.

    expansion: int = 4

    def __init__(
        self,
        embed_dim,
        kernel_size: int = 3,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MetaBlock4D, self).__init__()
        self.token_mixer = PoolMixer(kernel_size=kernel_size)
        self.feedforward = Sequential(
            Conv2d(
                embed_dim,
                embed_dim * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
                **factory_kwargs
            ),
            BatchNorm2d(embed_dim * self.expansion, **factory_kwargs),
            GELU(),
            Dropout(p=dropout),
            Conv2d(
                embed_dim * self.expansion,
                embed_dim,
                kernel_size=1,
                stride=1,
                bias=False,
                **factory_kwargs
            ),
            BatchNorm2d(embed_dim, **factory_kwargs),
            Dropout(p=dropout),
        )

    def forward(self, x):
        out = x
        out = out + self.token_mixer(out)
        out = out + self.feedforward(out)
        return out


class MetaBlock3D(Module):
    # "EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>.

    expansion: int = 4

    def __init__(self, embed_dim, dropout: float = 0.0, device=None, dtype=None):
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MetaBlock3D, self).__init__()
        self.token_mixer = MultiheadAttention(embed_dim, num_heads=8, **factory_kwargs)
        self.feedforward = Sequential(
            Linear(embed_dim, embed_dim * self.expansion, **factory_kwargs),
            GELU(),
            Dropout(p=dropout),
            Linear(embed_dim * self.expansion, embed_dim, **factory_kwargs),
            Dropout(p=dropout),
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x):
        out = x
        nrm = self.norm1(out)
        out = out + self.token_mixer(nrm, nrm, nrm)[0]
        out = out + self.feedforward(self.norm2(out))
        return out


class EfficientFormer(Module):
    def __init__(
        self,
        embed_dims: List[int],
        layers: List[int],
        num_classes: int = 1000,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(EfficientFormer, self).__init__()
        self.conv1 = Conv2d(
            3,
            int(embed_dims[0] / 2),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            **factory_kwargs
        )
        self.bn1 = BatchNorm2d(int(embed_dims[0] / 2), **factory_kwargs)
        self.conv2 = Conv2d(
            int(embed_dims[0] / 2),
            embed_dims[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            **factory_kwargs
        )
        self.bn2 = BatchNorm2d(embed_dims[0], **factory_kwargs)
        self.stage1 = self._make_layer(
            MetaBlock4D,
            embed_dims[0],
            embed_dims[1],
            layers[0],
            stride=2,
            **factory_kwargs
        )
        self.stage2 = self._make_layer(
            MetaBlock4D,
            embed_dims[1],
            embed_dims[2],
            layers[1],
            stride=2,
            **factory_kwargs
        )
        self.stage3 = self._make_layer(
            MetaBlock4D,
            embed_dims[2],
            embed_dims[3],
            layers[2],
            stride=2,
            **factory_kwargs
        )
        self.stage4 = self._make_layer(
            MetaBlock4D,
            embed_dims[3],
            embed_dims[3],
            layers[3],
            stride=1,
            **factory_kwargs
        )
        self.stage5 = self._make_layer(
            MetaBlock3D,
            embed_dims[3],
            embed_dims[3],
            layers[4],
            stride=1,
            **factory_kwargs
        )
        self.avgpool = AdaptiveAvgPool1d(1)
        self.fc = Linear(embed_dims[3], num_classes, **factory_kwargs)

    @staticmethod
    def _make_layer(
        blk: Type[Union[MetaBlock4D, MetaBlock3D]],
        planes: int,
        next_planes: int,
        blocks: int,
        stride: int = 1,
        device=None,
        dtype=None,
    ) -> Sequential:
        factory_kwargs = dict(device=device, dtype=dtype)
        layers = []
        for _ in range(blocks):
            layers.append(blk(planes, **factory_kwargs))

        if len(layers) == 0:
            layers.append(Identity())

        if stride != 1:
            layers.append(
                Sequential(
                    Conv2d(
                        planes,
                        next_planes,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                        **factory_kwargs
                    ),
                    BatchNorm2d(next_planes, **factory_kwargs),
                )
            )

        return Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # stage 4 MB4D

        # reshape once
        bsz, c, h, w = x.shape
        x = x.reshape(h * w, bsz, c)

        x = self.stage5(x)  # stage 4 MB3D

        # x = x.permute(1, 2, 0)
        # x = self.avgpool(x).squeeze(dim=-1)
        x = x.mean(dim=0)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientformer(
    pretrained: bool, embed_dim: List[int], layers: List[int], **kwargs: Any
):
    model = EfficientFormer(embed_dims=embed_dim, layers=layers, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def efficientformer_l1(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L1 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, [48, 96, 224, 448], [3, 2, 6, 3, 1], **kwargs)


def efficientformer_l3(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L3 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, [64, 128, 320, 512], [4, 4, 12, 3, 3], **kwargs)


def efficientformer_l7(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L7 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, [96, 192, 384, 768], [6, 6, 8, 0, 8], **kwargs)


if __name__ == "__main__":
    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    patches = torch.randn(8, 3, 224, 224)

    efficient_former_l1 = efficientformer_l1()
    efficient_former_l1.eval()

    efficient_former_l3 = efficientformer_l3()
    efficient_former_l3.eval()

    efficient_former_l7 = efficientformer_l7()
    efficient_former_l7.eval()

    mobilenet_v2 = models.mobilenet_v2()
    mobilenet_v2.eval()

    mobilenet_v3 = models.mobilenet_v3_small()
    mobilenet_v3.eval()

    for _ in range(10):
        efficient_former_l1(patches)
        efficient_former_l3(patches)
        efficient_former_l7(patches)
        mobilenet_v2(patches)
        mobilenet_v3(patches)
    print('warmup done.')

    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as p1:
            with record_function("inference_mobilenet_v2"):
                mobilenet_v2(patches)

        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as p2:
            with record_function("inference_mobilenet_v3_small"):
                mobilenet_v3(patches)

        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as p3:
            with record_function("inference_efficient_former_l1"):
                efficient_former_l1(patches)

        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as p4:
            with record_function("inference_efficient_former_l3"):
                efficient_former_l3(patches)

        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as p5:
            with record_function("inference_efficient_former_l7"):
                efficient_former_l7(patches)

    print(p1.key_averages().table(sort_by="cpu_time_total", row_limit=1))
    print(p2.key_averages().table(sort_by="cpu_time_total", row_limit=1))
    print(p3.key_averages().table(sort_by="cpu_time_total", row_limit=1))
    print(p4.key_averages().table(sort_by="cpu_time_total", row_limit=1))
    print(p5.key_averages().table(sort_by="cpu_time_total", row_limit=1))

    # breakpoint()
