import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt
from math import log2
import copy


def softmax(x):
    """
    softmax function
    """
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class SegmentCorrelation2(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        # factor is segment length
        super(SegmentCorrelation2, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag  # if True drop first else drop last

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)

        correlation_scores = torch.einsum("bmlhd,bnlhd->bhmn", seg_queries, seg_keys)  # (b, h, 5, 3)
        A = torch.softmax(scale * correlation_scores, dim=-1)  # (b, h, 5, 3)
        V = torch.einsum("bhmn,bnlhd->bmlhd", A, seg_values)  # (b, 5, l_s, h, d_v)

        V = V.reshape(B, -1, H, D_v)  # (b, l_q, h, d_v)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SegmentCorrelation3(nn.Module):
    # shift 1 segment
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        # factor is segment length
        super(SegmentCorrelation3, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag  # if True drop first else drop last

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_keys_pre = seg_keys[:, :-1, ...]  # (b, 2, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)
        seg_values_aft = seg_values[:, 1:, ...]  # (b, 2, l_s, h, d_v)

        correlation_scores = torch.einsum("bmlhd,bnlhd->bhmn", seg_queries, seg_keys_pre)  # (b, h, 5, 2)
        A = torch.softmax(scale * correlation_scores, dim=-1)  # (b, h, 5, 2)
        tmp_V = torch.einsum("bhmn,bnlhd->bmlhd", A, seg_values_aft)  # (b, 5, l_s, h, d_v)
        V = torch.roll(tmp_V, shifts=1, dims=1)

        V = V.reshape(B, -1, H, D_v)  # (b, l_q, h, d_v)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class MultiScaleAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(MultiScaleAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.attention_name = type(attention).__name__
        self.factor = attention.factor
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
            queries: (B, L_q, d_model)
            keys: (B, L_k, d_model)
            values: (B, L_v=L_k, d_model)
            attn_mask: (B, 1, L, L)
            return: (B, L_q, d_model)
        """
        B, L_q, _ = queries.shape
        _, L_k, _ = keys.shape
        H = self.n_heads
        L_min = min(L_q, L_k)
        scale_num = math.floor(log2(L_min // self.factor)) + 1
        attn_list = clones(self.inner_attention, scale_num-1)
        scale_weight = np.zeros(scale_num)
        for i in range(scale_num):
            scale_weight[i] = 1 / (2 ** i)
        scale_weight = scale_weight / sum(scale_weight)
        # scale_weight = softmax(scale_weight)
        # scale_weight[:] = 1 / scale_num

        queries = self.query_projection(queries).view(B, L_q, H, -1)  # (B, L_q, H, D_q)
        keys = self.key_projection(keys).view(B, L_k, H, -1)  # (B, L_k, H, D_k=D_q)
        values = self.value_projection(values).view(B, L_k, H, -1)  # (B, L_v=L_k, H, D_v)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out * scale_weight[0]

        head_flag = self.inner_attention.head_flag
        for i in range(1, scale_num):
            head_flag = not head_flag
            attn_list[i-1].factor = self.factor * (2 ** i)
            attn_list[i-1].head_flag = head_flag
            out1, _ = attn_list[i-1](
                queries,
                keys,
                values,
                attn_mask
            )
            out = out + out1 * scale_weight[i]
        out = out.view(B, L_q, -1)  # (B, L_q, n_heads * D_v)

        return self.out_projection(out), attn