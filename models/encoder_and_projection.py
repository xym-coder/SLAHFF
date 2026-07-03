# =================================================================================
# 文件: models/encoder_and_projection.py
# 请用以下完整代码，覆盖您现有的 models/encoder_and_projection.py 文件
# =================================================================================

import torch
from .mlp_head import MLPHead
from torch import nn
import torch.nn.functional as F

from .complexcnn import ComplexConv
from .attentions import MultiScaleDynamicWindowAttention

# ==================== 核心修改 1：导入新的带掩码的MSF模块 ====================
from .msf_module import MSF_Module_With_Mask
# =============================================================================


class AttentionFusionModule(nn.Module):
    # ... (此部分代码保持不变)
    def __init__(self, feature_dim_iq, feature_dim_af):
        super(AttentionFusionModule, self).__init__()
        self.iq_attention = nn.Linear(feature_dim_iq, feature_dim_iq)
        self.af_attention = nn.Linear(feature_dim_af, feature_dim_af)
        self.fusion_weight = nn.Parameter(torch.ones(2))

    def forward(self, iq_feature, af_feature):
        iq_score = torch.sigmoid(self.iq_attention(iq_feature))
        af_score = torch.sigmoid(self.af_attention(af_feature))
        weighted_iq_feature = iq_feature * iq_score
        weighted_af_feature = af_feature * af_score
        fusion_weights = F.softmax(self.fusion_weight, dim=0)
        fused_feature = torch.cat([weighted_iq_feature, weighted_af_feature], dim=1)
        return fused_feature, fusion_weights


class SqueezeExciteBlock(nn.Module):
    # ... (此部分代码保持不变)
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        out = self.avg_pool(x).squeeze(-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.unsqueeze(-1)
        return x * out


class Encoder_and_projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder_and_projection, self).__init__()

        # --- CNN特征提取器部分，保持完全不变 ---
        self.conv1 = ComplexConv(in_channels=1, out_channels=64, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.se1 = SqueezeExciteBlock(128)
        self.conv2 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn2 = nn.BatchNorm1d(num_features=128);
        self.se2 = SqueezeExciteBlock(128)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn3 = nn.BatchNorm1d(num_features=128);
        self.se3 = SqueezeExciteBlock(128)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn4 = nn.BatchNorm1d(num_features=128);
        self.se4 = SqueezeExciteBlock(128)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn5 = nn.BatchNorm1d(num_features=128);
        self.se5 = SqueezeExciteBlock(128)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn6 = nn.BatchNorm1d(num_features=128);
        self.se6 = SqueezeExciteBlock(128)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn7 = nn.BatchNorm1d(num_features=128);
        self.se7 = SqueezeExciteBlock(128)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn8 = nn.BatchNorm1d(num_features=128);
        self.se8 = SqueezeExciteBlock(128)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=2);
        self.bn9 = nn.BatchNorm1d(num_features=128);
        self.se9 = SqueezeExciteBlock(128)

        # ==================== 核心修改 2：实例化新的带掩码的MSF模块 ====================
        # CNN的输出通道是128，所以MSF的输入输出通道也设为128
        self.msf_module = MSF_Module_With_Mask(in_channels=128, out_channels=128)
        # =============================================================================

        # --- MDWA 模块定义，保持不变 ---
        window_sizes = [2, 4, 7]
        self.norm1 = nn.LayerNorm([128, 7])
        self.multi_scale_attention = MultiScaleDynamicWindowAttention(
            d_model=128,
            nhead=8,
            window_sizes=window_sizes,
            dropout=0.1
        )

        # --- 后续层保持不变 ---
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(1024)
        self.projetion = MLPHead(in_channels=1024, **kwargs["projection_head"])
        self.fusion_module = AttentionFusionModule(feature_dim_iq=1024, feature_dim_af=20)

    def forward(self, x):
        x_input = x[:, :, :4800]
        x_extra = x[:, 0, -20:]

        # 1. 通过CNN特征提取器 (保持不变)
        x = F.relu(self.bn1(self.conv1(x_input)));
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)));
        x = self.se2(x)
        x = F.relu(self.bn3(self.conv3(x)));
        x = self.se3(x)
        x = F.relu(self.bn4(self.conv4(x)));
        x = self.se4(x)
        x = F.relu(self.bn5(self.conv5(x)));
        x = self.se5(x)
        x = F.relu(self.bn6(self.conv6(x)));
        x = self.se6(x)
        x = F.relu(self.bn7(self.conv7(x)));
        x = self.se7(x)
        x = F.relu(self.bn8(self.conv8(x)));
        x = self.se8(x)
        x = F.relu(self.bn9(self.conv9(x)));
        x = self.se9(x)
        # CNN输出x的形状是 (B, 128, 7)

        # 2. 通过改造后的 MSF 模块 (MSF_Module_With_Mask)
        x = self.msf_module(x)
        # 输出形状仍然是 (B, 128, 7)，但特征已经被内部的掩码提纯过

        # 3. 应用MDWA模块
        x_shortcut = x
        x = self.norm1(x)
        x = self.multi_scale_attention(x)
        x = x + x_shortcut

        # 4. 展平和全连接层 (保持不变)
        x = self.flatten(x)
        x = self.fc(x)
        embedding = F.relu(x)
        project_out = self.projetion(embedding)

        # 5. 应用SAFM模块进行特征融合 (保持不变)
        embedding_new = x_extra
        fused_embedding, fusion_weights = self.fusion_module(embedding, embedding_new)

        return fused_embedding, project_out, embedding, embedding_new

