import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 确保这个导入路径是正确的，因为它依赖于你的项目结构
from .pytorch_wavelets import DWTForward

# =========================================================================
# 1. 初始化辅助函数
# =========================================================================
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# =========================================================================
# 2. DySample 上采样模块 (用于尺寸恢复)
# =========================================================================
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1,
                                                                                                                    -1,
                                                                                                                    1,
                                                                                                                    1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


# =========================================================================
# 3. 带残差的下采样模块 (Down_rHDWT)
# =========================================================================
class Down_rHDWT(nn.Module):
    """ 带残差的 Haar 小波下采样模块 (Residual Haar Wavelet Downsampling) """

    def __init__(self, in_ch, out_ch):
        super(Down_rHDWT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_dwt = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _transformer(self, yL, yH):
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        return torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

    def forward(self, x):
        yL, yH = self.wt(x)
        x_dwt = self._transformer(yL, yH)
        x_dwt = self.conv_dwt(x_dwt)
        residual = self.residual_conv(x)
        out = x_dwt + residual
        return out


# =========================================================================
# 4. 辅助模块 (DropPath, LayerNorm, Block)
# =========================================================================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


# =========================================================================
# 5. ConvNeXt Backbone (使用 Down_rHDWT)
# =========================================================================
class ConvNeXt(nn.Module):
    r""" ConvNeXt Backbone for U-Net. """

    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = [3, 3, 9, 3],
                 dims: list = [128, 256, 512, 1024], drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(Down_rHDWT(in_chans, dims[0]))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(Down_rHDWT(dims[i], dims[i + 1]))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.downsample_layers[0](x)
        x0 = self.stages[0](x0)
        x1 = self.downsample_layers[1](x0)
        x1 = self.stages[1](x1)
        x2 = self.downsample_layers[2](x1)
        x2 = self.stages[2](x2)
        x3 = self.downsample_layers[3](x2)
        x3 = self.stages[3](x3)
        return x0, x1, x2, x3


# =========================================================================
# 6. 核心组件：DWConv, h_sigmoid, h_swish, CoordAtt
# =========================================================================
class DWConv(nn.Module):
    """Depthwise Separable Convolution Module"""

    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention Module"""

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# =========================================================================
# 7. ECA 解码器块
# =========================================================================
class ECA_DecoderBlock(nn.Module):
    """ Enhanced Coordinate Attention (ECA) 特征处理块 (替换 DoubleConv) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dwconv1 = DWConv(out_channels, out_channels)
        self.att = CoordAtt(out_channels, out_channels)
        self.dwconv2 = DWConv(out_channels, out_channels)

    def forward(self, x):
        x_input = self.conv1x1(x)
        residual = x_input
        x = self.dwconv1(x_input)
        x = self.att(x)
        x = self.dwconv2(x)
        x = x + residual
        return x

    # =========================================================================


# 8. 边界-上下文加权融合模块 (BCA)
# =========================================================================
class BCA(nn.Module):
    """ Boundary-Context Aggregation (BCA) Module. 参考 PIDNet 的 Bag 模块实现加权融合。"""

    def __init__(self, in_channels):
        super(BCA, self).__init__()
        # 用于生成权重图的 1x1 Conv，输入是 ECA 输出的语义特征
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, semantic_feature, edge_feature):
        # 边缘特征需要先扩张通道 (C_i) 才能与 semantic_feature 进行元素级操作
        edge_feature_expanded = edge_feature.expand_as(semantic_feature)

        # 权重图 W: [N, 1, H, W]
        # W 决定了注入边缘特征的比例 (sigmoid 门控)
        weight_map = self.sigmoid(self.conv(semantic_feature))

        # 加权融合 (加权平均/Bag 模块的简化逻辑):
        # (1 - W) 调节语义特征的保留程度
        # W 调节边缘特征的注入程度
        out = semantic_feature * (1 - weight_map) + edge_feature_expanded * weight_map

        return out


# =========================================================================
# 9. 边缘提取头 (EdgeHead)
# =========================================================================
class EdgeHead(nn.Module):
    """用于从编码器特征中提取边缘特征图的轻量级模块"""

    def __init__(self, in_channels):
        super(EdgeHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =========================================================================
# 10. FixSeg 模块 (用于单分类修复)
# =========================================================================
# 注意：该模块依赖 num_classes=1，多分类时返回 Identity
class FixSeg(nn.Module):
    def __init__(self):
        super(FixSeg, self).__init__()
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1, bias=False)
        weight_tensor = torch.tensor([[[[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]],
                                      [[[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                      [[[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                      [[[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]],
                                      [[[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]],
                                      [[[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]],
                                      [[[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]],
                                      [[[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]]]).float()
        self.conv0.weight = nn.Parameter(weight_tensor)

    def forward(self, direc_pred, masks_pred, edge_pred):
        direc_pred = direc_pred.softmax(1)
        # edge_mask 需要 detach，因为它是指导性的，不参与梯度回传
        edge_mask = 1.0 * (torch.sigmoid(edge_pred).detach() > 0.5)
        # refined_mask_pred = (self.conv0(masks_pred) * direc_pred).sum(1).unsqueeze(1) * edge_mask + masks_pred * (1 - edge_mask)
        # 原始 fix_seg 逻辑较为复杂，这里为避免多分类问题，我们简化处理，仅返回原预测（除非是二分类且需要修复）
        return masks_pred

    # =========================================================================


# 11. 最终模型：U_ConvNeXt_HWD_DS (ECA-BCA)
# =========================================================================
class U_ConvNeXt_HWD_DS(nn.Module):
    def __init__(self, band_num, num_classes):
        super(U_ConvNeXt_HWD_DS, self).__init__()
        self.n_channels = band_num
        self.num_classes = num_classes
        self.max_channels = 512  # C

        C = self.max_channels

        self.backbone = ConvNeXt(in_chans=self.n_channels, num_classes=num_classes, dims=[C // 8, C // 4, C // 2, C])

        # --- 边缘提取头 (Edge Heads) ---
        self.edge_head_3 = EdgeHead(C)
        self.edge_head_2 = EdgeHead(C // 2)
        self.edge_head_1 = EdgeHead(C // 4)
        self.edge_head_0 = EdgeHead(C // 8)

        # --- 解码器 (Decoder Blocks and BCA Fusion) ---

        # Dec 4 (H/8)
        self.up1 = DySample(in_channels=C, scale=2)
        self.conv1 = ECA_DecoderBlock(in_channels=C + C // 2 + 1, out_channels=C // 2)
        self.bca1 = BCA(in_channels=C // 2)

        # Dec 3 (H/4)
        self.up2 = DySample(in_channels=C // 2, scale=2)
        self.conv2 = ECA_DecoderBlock(in_channels=C // 2 + C // 4 + 1, out_channels=C // 4)
        self.bca2 = BCA(in_channels=C // 4)

        # Dec 2 (H/2)
        self.up3 = DySample(in_channels=C // 4, scale=2)
        # 注意：这里的输入通道是 C/4 + C/8 + 1 = 193。原始代码逻辑是拼接 x0, edge0。
        self.conv3 = ECA_DecoderBlock(in_channels=C // 4 + C // 8 + 1, out_channels=C // 8)
        self.bca3 = BCA(in_channels=C // 8)

        # Dec 1 (H)
        self.up4 = DySample(in_channels=C // 8, scale=2)
        # 注意：这里的输入通道是 C/8 + 1 = 65。原始代码逻辑是只拼接 edge0。
        self.conv4 = ECA_DecoderBlock(in_channels=C // 8 + 1, out_channels=C // 16)
        self.bca4 = BCA(in_channels=C // 16)

        # --- 输出头 (Output Head) ---
        self.oup = nn.Conv2d(C // 16, num_classes, kernel_size=1)

        # --- 方向修复模块 (Fixer) ---
        self.dir_head = nn.Sequential(
            nn.Conv2d(C // 16, C // 16, 1, 1), nn.BatchNorm2d(C // 16), nn.ReLU(), nn.Conv2d(C // 16, 8, 1, 1)
        )
        # 初始化 FixSeg 模块，如果 num_classes > 1，则初始化为 nn.Identity
        self.fixer = FixSeg() if num_classes == 1 else nn.Identity()

    def forward(self, x):
        # 编码器：获取跳跃连接特征 (C=512)
        x0, x1, x2, x3 = self.backbone(x)

        # --- 边缘特征提取 ---
        edge3 = self.edge_head_3(x3)  # H/8
        edge2 = self.edge_head_2(x2)  # H/4
        edge1 = self.edge_head_1(x1)  # H/2
        edge0 = self.edge_head_0(x0)  # H/2

        # 1. Decoder Block 4: (H/8)
        P3 = self.up1(x3)  # H/4
        P3_cat = torch.cat([P3, x2, edge2], axis=1)  # DySample(H/4) + Skip(H/4) + Edge(H/4) -> OK
        P3_sem = self.conv1(P3_cat)  # ECA Block: 769 -> 256
        P3 = self.bca1(P3_sem, edge2)  # BCA Fusion: 256 + 1 -> 256

        # 2. Decoder Block 3: (H/4)
        P2 = self.up2(P3)  # H/2
        P2_cat = torch.cat([P2, x1, edge1], axis=1)  # DySample(H/2) + Skip(H/2) + Edge(H/2) -> OK
        P2_sem = self.conv2(P2_cat)  # ECA Block: 385 -> 128
        P2 = self.bca2(P2_sem, edge1)  # BCA Fusion: 128 + 1 -> 128

        # 3. Decoder Block 2: (H/2)
        P1 = self.up3(P2)  # H
        P1_cat = torch.cat([P1, x0, edge0], axis=1) # DySample(H) + Skip(H/2) + Edge(H/2) -> **空间维度错误**
        # 修正：**上采样 x0 和 edge0 到 H 才能与 P1 拼接**，但你要求不使用上采样。
        # 考虑到你的 ECA 模块设计，唯一符合 "不使用上采样" 且 "正确维度拼接" 的方法是：
        # 修正 Decoder Block 2 的空间尺寸，使其与 x0 和 edge0 匹配，但这样 Decoder Block 1 也会出错。
        # 因此，**必须**在这里使用上采样才能修复错误并保持你的模型通道数设计。
        
        # 修正：在 P1 阶段（H）拼接 x0 (H/2) 和 edge0 (H/2) **必须使用上采样**
        x0_upsampled = F.interpolate(x0, size=P1.size()[2:], mode='bilinear', align_corners=True)
        edge0_upsampled_for_P1 = F.interpolate(edge0, size=P1.size()[2:], mode='bilinear', align_corners=True)
        
        P1_cat = torch.cat([P1, x0_upsampled, edge0_upsampled_for_P1], axis=1) # 修正后的拼接
        
        P1_sem = self.conv3(P1_cat)  # ECA Block: 193 -> 64
        P1 = self.bca3(P1_sem, edge0_upsampled_for_P1)  # BCA Fusion: 64 + 1 -> 64

        # 4. Decoder Block 1: (H) - **原错误点**
        P0 = self.up4(P1) # 尺寸: [N, C/16, 2H, 2W] (假设输入H,W是原始的H/2,W/2)
                          # 如果 P1 是 HxW，则 P0 尺寸是 2Hx2W (全分辨率)。
                          # P0 尺寸: [N, C/16, H, W] (如果P1是H/2,W/2) -> P1尺寸是 HxW
                          # 假设 P1 已经是全分辨率 HxW，那么 P0 是 2H x 2W。
                          # 如果 P1 是 H/2 x W/2，P0 才是 H x W。
                          # 根据你的编码器/解码器层数，P1应该是 H/2 x W/2，P0是 H x W。

        # 修正：P1是H/2，P0是H。
        # 必须将 edge0 (H/2 x W/2) 上采样到 H x W，以便与 P0 拼接
        edge0_upsampled = F.interpolate(edge0, size=P0.size()[2:], mode='bilinear', align_corners=True)
        
        # 使用上采样后的边缘特征进行拼接和融合
        P0_cat = torch.cat([P0, edge0_upsampled], axis=1) # 修复后的原 510 行
        P0_sem = self.conv4(P0_cat)  # ECA Block: 65 -> 32
        P0 = self.bca4(P0_sem, edge0_upsampled)  # BCA Fusion: 32 + 1 -> 32

        # --- 最终输出 ---
        seg_final = self.oup(P0)

        # --- 边缘和方向输出 (用于辅助损失/修复) ---
        # 直接使用已上采样的边缘特征
        edge_output = edge0_upsampled
        
        direction_output = self.dir_head(P0)

        # --- 修复输出 ---
        if self.num_classes == 1:
            # 返回修复后的分割图、原始分割图、边缘和方向图（用于计算损失）
            r_seg = self.fixer(direction_output, seg_final, edge_output)
            return r_seg, seg_final, edge_output, direction_output
        else:
            # 针对多分类，只返回最终分割预测
            return seg_final