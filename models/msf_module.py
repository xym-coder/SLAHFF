# =================================================================================
# 文件: models/msf_module.py
# 【最终正确版】请用以下完整代码，覆盖您现有的 models/msf_module.py 文件
# =================================================================================

import torch
from torch import nn
import torch.nn.functional as F


# ------------------- 新增的简化版门控掩码模块 (GatedMaskModule) -------------------
class GatedMaskModule(nn.Module):
    """
    简化版的门控掩码模块，严格按照手绘图实现。
    它接收一个特征图，通过三个并行的1x1卷积生成一个动态掩码，并应用到原始特征上。
    """

    def __init__(self, channels):
        super(GatedMaskModule, self).__init__()
        self.conv_sigmoid = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv_tanh = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv_filter = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        # x 的形状: (B, C, L), 例如 (B, 128, 7)

        gate_sigmoid = torch.sigmoid(self.conv_sigmoid(x))
        content_tanh = torch.tanh(self.conv_tanh(x))
        gated_feature = gate_sigmoid * content_tanh

        dynamic_filter = self.conv_filter(gated_feature)

        # 将滤波器应用到原始输入上 (Element-wise multiplication)
        return x * dynamic_filter


# ------------------- 改造后的 MSF 模块 (恢复了相加逻辑) -------------------
class MSF_Module_With_Mask(nn.Module):
    """
    集成了门控掩码的全新MSF模块。
    严格按照 "并行分支 -> 独立掩码 -> 相加 -> 融合" 的最终正确逻辑实现。
    """

    def __init__(self, in_channels, out_channels):
        super(MSF_Module_With_Mask, self).__init__()

        # --- 并行分支定义 ---
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        # --- 为每个分支实例化独立的门控掩码模块 ---
        self.mask_branch1 = GatedMaskModule(channels=out_channels)
        self.mask_branch2 = GatedMaskModule(channels=out_channels)

        # --- 融合层定义 (恢复了相加逻辑) ---
        # 输入通道数是 out_channels，因为两个分支是相加而不是拼接
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # 1. 分别处理两个并行分支
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)

        # 2. 在融合前，对每个分支的输出独立进行掩码提纯
        refined_feat1 = self.mask_branch1(feat1)
        refined_feat2 = self.mask_branch2(feat2)

        # 3. 将经过提纯后的两个分支特征进行相加 (Element-wise Addition)
        out = refined_feat1 + refined_feat2

        # 4. 使用1x1卷积进行最终融合
        out = self.conv_fusion(out)

        return out

