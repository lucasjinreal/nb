# NB

Nenural network Blocks (aka: **NB**, or neural network builder). This library provides massive fancy blocks for you for quick import to build your powerful. Some SOTA tricks and connections such as CSP, ASFF, Attention, BaseConv, Hardswish, Mish all included for quick prototype your model. This is an **Arsenal for deeplearning forge**.

**nb** is an idea comes from engineering, we build model with some common blocks, we exploring new ideas with SOTA tricks, but all those thing can be gathered into one single place, and for model quick design and prototyping.

this project is under construct for now, I will update it quickly once I found some new blocks that really works in model. Also, every single updated block will be recorded in updates.



## Install

**nb** can be installed from PIP, remember the name is `nbnb`:

```
sudo pip3 install nbnb
```



## Usage

Here is an example of using NB to build YoloV5! 

**updates**: We have another YoloV5-ASFF version added in example!

```python
import torch
from torch import nn
from nb.torch.blocks.bottleneck_blocks import SimBottleneckCSP
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.head_blocks import SPP
from nb.torch.blocks.conv_blocks import ConvBase
from nb.torch.utils import device

class YoloV5(nn.Module):

    def __init__(self, num_cls=80, ch=3, anchors=None):
        super(YoloV5, self).__init__()
        assert anchors != None, 'anchor must be provided'

        # divid by
        cd = 2
        wd = 3

        self.focus = Focus(ch, 64//cd)
        self.conv1 = ConvBase(64//cd, 128//cd, 3, 2)
        self.csp1 = SimBottleneckCSP(128//cd, 128//cd, n=3//wd)
        self.conv2 = ConvBase(128//cd, 256//cd, 3, 2)
        self.csp2 = SimBottleneckCSP(256//cd, 256//cd, n=9//wd)
        self.conv3 = ConvBase(256//cd, 512//cd, 3, 2)
        self.csp3 = SimBottleneckCSP(512//cd, 512//cd, n=9//wd)
        self.conv4 = ConvBase(512//cd, 1024//cd, 3, 2)
        self.spp = SPP(1024//cd, 1024//cd)
        self.csp4 = SimBottleneckCSP(1024//cd, 1024//cd, n=3//wd, shortcut=False)

        # PANet
        self.conv5 = ConvBase(1024//cd, 512//cd)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = SimBottleneckCSP(1024//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv6 = ConvBase(512//cd, 256//cd)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = SimBottleneckCSP(512//cd, 256//cd, n=3//wd, shortcut=False)

        self.conv7 = ConvBase(256//cd, 256//cd, 3, 2)
        self.csp7 = SimBottleneckCSP(512//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv8 = ConvBase(512//cd, 512//cd, 3, 2)
        self.csp8 = SimBottleneckCSP(512//cd, 1024//cd, n=3//wd, shortcut=False)

    def _build_backbone(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x_p3 = self.conv2(x)  # P3
        x = self.csp2(x_p3)
        x_p4 = self.conv3(x)  # P4
        x = self.csp3(x_p4)
        x_p5 = self.conv4(x)  # P5
        x = self.spp(x_p5)
        x = self.csp4(x)
        return x_p3, x_p4, x_p5, x

    def _build_head(self, p3, p4, p5, feas):
        h_p5 = self.conv5(feas)  # head P5
        x = self.up1(h_p5)
        x_concat = torch.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = torch.cat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = torch.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_large = self.csp8(x)
        return x_small, x_medium, x_large

    def forward(self, x):
        p3, p4, p5, feas = self._build_backbone(x)
        xs, xm, xl = self._build_head(p3, p4, p5, feas)
        return xs, xm, xl
```

A simple example to build a layer of conv:

```python
from nb.torch.base.conv_block import ConvBase
a = ConvBase(128, 256, 3, 1, 2, norm_cfg=dict(type="BN"), act_cfg=dict(type="Hardswish"))
```
Be note that, the reason for us using `cfg` to specific norm and activation is for users dynamically switch their configuration of model in yaml format rather than hard code it.

A simple example of using GhostNet:

```python
from nb.torch.backbones.ghostnet import GhostNet

m = GhostNet(num_classes=8)

# if you want FPN output
m = GhostNet(fpn_levels=[4, 5, 6])
```

A simple example of using MobilenetV3:

```python
from nb.torch.backbones.mobilenetv3_new import MobilenetV3_Small
```





## Updates

- **2021.03.16** Added some blocks used inside Scaled-YoloV4 (P5,P6,P7). List are:
  
  - `HarDBlock`;
  - `SPPCSP`;
  - `VoVCSP`;
  
  You can using these blocks to stack your model now.
  
  ```python
  from nb.torch.blocks.csp_blocks import VoVCSP
  ```
  
  
  
- **2021.01.22**: Adding Mish activation function. You can call it in your model using the following code:
  
  ```python
  from nb.torch.base import build_activation_layer
  act = build_activation_layer(act_cfg=dict(type='Mish'))
  ```
  
- **2021.01.22**: Adding Triplet Attention Mechanism. You can plug it in any of your conv net blocks using the following code:
  
  ```python
  from nb.torch.blocks.attention_blocks import TripletAttention
  att_mechanism = TripletAttention()
  rand_tensor = torch.rand(1,3,32,32)
  output = att_mechanism(rand_tensor)
  ```
TripletAttention is a shape preserving tensor which expects a 4-dimensional input (B,C,H,W) and outputs a 4-dimensional output of the same shape (B,C,H,W).
  
- **2021.01.14**: Adding SiLU introduced from pytorch 1.7. And now you can build a activation layer by using:

  ```python
  from nb.torch.base import build_activation_layer
  act = build_activation_layer(act_cfg=dict(type='SiLU'))
  ```

  Also PANet module also provided now. BiFPN is on the way. We will also provide more examples on how to using it!

- **2020.09.28**: ASFF module added inside **nb**. We have a ASFF design version of YoloV5 now! Some experiment will add here once we confirm ASFF module enhance the model performance.

- **2020.09.22**: New backbone of `Ghostnet` and `MobilenetV3` included. Both of them can be used to replace any of your application's backbone.

- **2020.09.14**: We release a primary version of 0.04, which you can build a simple YoloV5 with **nb** easily!

  ```shell
  pip install nbnb
  ```
  
- **2020.09.12**: New backbone SpineNet added:

  SpineNet is a backbone model specific for detection, it's a backbone but can do FPN's thing!! More info pls reference google's paper [link](https://ai.googleblog.com/2020/06/spinenet-novel-architecture-for-object.html).
  
  ```python
  from nb.torch.bakbones.spinenet import SpineNet
  
  model = SpineNet()
  ```
  
- **2020.09.11**: New added blocks:

  ```
  resnet.Bottleneck
  resnet.BasicBlock
  
  ConvBase
  ```





## Support Matrix

We list all `conv` and `block` support in **nb** here:

- `conv`:
  - Conv
  - ConvWS: https://arxiv.org/pdf/1903.10520.pdf
  - ...
- `Blocks`:
  - CSPBlock: 



## Copyright

@Lucas Jin all rights reserved.
