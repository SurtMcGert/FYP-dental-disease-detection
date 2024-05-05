from pickle import FALSE
from torch import nn
import torch
from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvBNSiLU, \
    MBLABlock, ConvBNHS, Lite_EffiBlockS2, Lite_EffiBlockS1
import math


class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class EfficientRep6(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False,
        generate_heat_maps=False,
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.generate_heat_maps = generate_heat_maps

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            )
        )

        channel_merge_layer = SimSPPF if not cspsppf else SimCSPSPPF

        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                block=block,
            ),
            # channel_merge_layer(
            #     in_channels=channels_list[5],
            #     out_channels=channels_list[5],
            #     kernel_size=5
            # )
        )

        self.num_of_attention_heads = 4
        # self.attention_layer = MultiHeadedSelfAttention(
        #     channels_list[5], self.num_of_attention_heads)

        self.attention_block = nn.Sequential(
            MultiHeadedSelfAttention(
                in_dim=channels_list[5],
                num_heads=self.num_of_attention_heads
            ),
            channel_merge_layer(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        x = self.attention_block(x)
        outputs.append(x)

        attention_weights = None

        if (self.training):
            return tuple(outputs)
        else:
            return tuple(outputs), attention_weights


class MultiHeadedSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, num_heads):
        assert in_dim % num_heads == 0, f"input_dim ({in_dim}) must be divisible by num_heads ({num_heads})"
        assert isinstance(
            in_dim, int), f"input_dim ({in_dim}) must be an integer"
        super(MultiHeadedSelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.num_heads = num_heads
        self.k_dim = in_dim / num_heads

        self.conv_q = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // self.num_heads, kernel_size=1)
        self.conv_k = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // self.num_heads, kernel_size=1)
        self.conv_v = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q = q.view(batch_size, -1, width * height)
        k = k.view(batch_size, -1, width * height)
        v = v.view(batch_size, -1, width * height)

        # print(f"proj_q: {q.size()}")
        # print(f"proj_k: {k.size()}")
        # print(f"proj_v: {v.size()}")

        q_heads = q.view(batch_size, self.num_heads,
                         -1, height * width)
        k_heads = k.view(batch_size, self.num_heads,
                         -1, height * width)
        v_heads = v.view(batch_size, self.num_heads,
                         -1, height * width)

        # print(f"q_heads: {q_heads.size()}")
        # print(f"k_heads: {k_heads.size()}")
        # print(f"v_heads: {v_heads.size()}")

        attention_scores = torch.matmul(
            q_heads.transpose(-2, -1), k_heads) / math.sqrt(self.k_dim)

        # print(f"attention_scores: {attention_scores.size()}")

        attention_weights = self.softmax(attention_scores)

        # print(f"attention_weights: {attention_weights.size()}")

        out = torch.matmul(v_heads, attention_weights.transpose(-2, -1))
        # print(f"out: {out.size()}")
        out = out.view(batch_size, C, width, height)

        out = (self.gamma * out) + x
        return out


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1) / 2,
        fuse_P2=False,
        cspsppf=False,
        stage_block_type="BepC3"
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone_P6(nn.Module):
    """
    CSPBepBackbone+P6 module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1) / 2,
        fuse_P2=False,
        cspsppf=False,
        stage_block_type="BepC3"
    ):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.fuse_P2 = fuse_P2

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
        )
        self.ERBlock_6 = nn.Sequential(
            block(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[5],
                out_channels=channels_list[5],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)


class Lite_EffiBackbone(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_repeat=[1, 3, 7, 3]
                 ):
        super().__init__()
        out_channels[0] = 24
        self.conv_0 = ConvBNHS(in_channels=in_channels,
                               out_channels=out_channels[0],
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.lite_effiblock_1 = self.build_block(num_repeat[0],
                                                 out_channels[0],
                                                 mid_channels[1],
                                                 out_channels[1])

        self.lite_effiblock_2 = self.build_block(num_repeat[1],
                                                 out_channels[1],
                                                 mid_channels[2],
                                                 out_channels[2])

        self.lite_effiblock_3 = self.build_block(num_repeat[2],
                                                 out_channels[2],
                                                 mid_channels[3],
                                                 out_channels[3])

        self.lite_effiblock_4 = self.build_block(num_repeat[3],
                                                 out_channels[3],
                                                 mid_channels[4],
                                                 out_channels[4])

    def forward(self, x):
        outputs = []
        x = self.conv_0(x)
        x = self.lite_effiblock_1(x)
        x = self.lite_effiblock_2(x)
        outputs.append(x)
        x = self.lite_effiblock_3(x)
        outputs.append(x)
        x = self.lite_effiblock_4(x)
        outputs.append(x)
        return tuple(outputs)

    @staticmethod
    def build_block(num_repeat, in_channels, mid_channels, out_channels):
        block_list = nn.Sequential()
        for i in range(num_repeat):
            if i == 0:
                block = Lite_EffiBlockS2(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=2)
            else:
                block = Lite_EffiBlockS1(
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=1)
            block_list.add_module(str(i), block)
        return block_list
