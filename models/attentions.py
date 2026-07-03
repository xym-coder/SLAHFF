import torch
from torch import nn
import torch.nn.functional as F
import math


# 这里是线性注意力机制的核心，我们使用一个简单的实现
# 基于"Performer"论文中的思想，使用一个正态随机投影矩阵来近似 softmax
# 这比标准的注意力计算效率高得多
class LinearAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(LinearAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, L, D = q.shape  # (batch_size, sequence_length, d_model)

        q = self.q_proj(q).reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)  # B, h, L, d_h
        k = self.k_proj(k).reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)  # B, h, L, d_h
        v = self.v_proj(v).reshape(B, L, self.nhead, self.head_dim).transpose(1, 2)  # B, h, L, d_h

        # 将 q, k 映射到正空间，实现线性注意力
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 核心计算：(K^T * V) * Q
        kv = torch.einsum('b h l d, b h l c -> b h d c', k, v)
        qk_sum = torch.einsum('b h d c, b h l d -> b h l c', kv, q)

        # 归一化
        denom = torch.einsum('b h l d, b h d -> b h l', q, k.sum(dim=2))
        attn_output = qk_sum / (denom.unsqueeze(-1) + 1e-8)

        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        attn_output = self.out_proj(attn_output)

        return attn_output


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size, dropout=0.1):
        super(SlidingWindowAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.head_dim = d_model // nhead

        self.attention = LinearAttention(d_model, nhead, dropout)

    def pad(self, x, window_size):
        # 确保序列长度是窗口大小的整数倍，以便进行分块
        seq_len = x.size(1)
        padding_len = (window_size - seq_len % window_size) % window_size
        if padding_len > 0:
            x = F.pad(x, (0, 0, 0, padding_len))
        return x, padding_len

    def forward(self, x):
        # x 维度: (batch_size, channels, seq_len)
        B, C, S = x.shape
        x = x.permute(0, 2, 1)  # 变为 (B, S, C)

        # 对输入序列进行分块和填充
        x_padded, padding_len = self.pad(x, self.window_size)
        _, S_padded, _ = x_padded.shape

        # 分割为窗口块
        x_blocks = x_padded.reshape(B, S_padded // self.window_size, self.window_size, C)

        # 对每个窗口块应用线性注意力
        output_blocks = []
        for i in range(x_blocks.size(1)):
            block = x_blocks[:, i, :, :]  # (B, window_size, C)
            output_blocks.append(self.attention(block, block, block))

        # 将处理后的窗口块重新拼接
        output = torch.cat(output_blocks, dim=1)

        # 移除填充
        if padding_len > 0:
            output = output[:, :S, :]

        output = output.permute(0, 2, 1)  # 变回 (B, C, S)

        return output


# ====================================================================================
# ===== 将以下代码完整复制到 attentions.py 文件末尾 =====
# ====================================================================================

class MultiScaleDynamicWindowAttention(nn.Module):
    """
    多尺度动态窗口注意力 (MDWA) 模块。
    这是一个全新的创新模块，它并行地使用多个不同窗口大小的SlidingWindowAttention（作为基础构建块），
    并自适应地融合它们的输出，以同时捕捉信号的局部、中程和全局特征。
    """

    def __init__(self, d_model, nhead, window_sizes, dropout=0.1):
        super(MultiScaleDynamicWindowAttention, self).__init__()
        self.d_model = d_model
        self.window_sizes = window_sizes

        # 1. 为每个定义的窗口大小创建一个并行的注意力分支
        #    这里巧妙地复用了我们已有的SlidingWindowAttention作为基础单元。
        self.attention_branches = nn.ModuleList([
            SlidingWindowAttention(d_model, nhead, ws, dropout) for ws in window_sizes
        ])

        # 2. 设计自适应融合层
        #    输入维度是 d_model * 分支数量 (因为我们在通道维度拼接)
        #    输出维度是 d_model，将多尺度的信息智能地融合回原始特征维度。
        self.fusion_layer = nn.Linear(d_model * len(window_sizes), d_model)

        # 3. 添加一个LayerNorm层来稳定融合后的特征
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x 的输入形状: (Batch, Channels, Seq_len), 例如 (B, 128, 7)

        # 1. 并行计算每个尺度分支的输出
        branch_outputs = [attn_branch(x) for attn_branch in self.attention_branches]

        # 2. 将所有分支的输出在“通道”维度上拼接(concatenate)
        #    例如，3个分支的输出都是(B, 128, 7)，拼接后变为 (B, 128 * 3, 7)
        concatenated_output = torch.cat(branch_outputs, dim=1)

        # 3. 调整维度以适应全连接的融合层
        #    (B, 128 * 3, 7) -> (B, 7, 128 * 3)
        concatenated_output = concatenated_output.permute(0, 2, 1)

        # 4. 通过融合层进行自适应融合，并进行归一化
        #    (B, 7, 128 * 3) -> (B, 7, 128)
        fused_output = self.fusion_layer(concatenated_output)
        fused_output = self.fusion_norm(fused_output)

        # 5. 恢复为原始的 (B, C, S) 形状，以便进行残差连接等后续操作
        #    (B, 7, 128) -> (B, 128, 7)
        fused_output = fused_output.permute(0, 2, 1)

        return fused_output
