# Multi-head attention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
        D : dimentions of K,V,Q
        num_heads: number of attention heads
    """
    def __init__(self, D, num_heads):
        self.num_heads = num_heads
        self.D_head = int(D / num_heads)

class DotProductAttention(nn.Module):
    """
        Compute the dot products of the query with all values
    """
    def forward(self, Q, V):
        pass

class Transformer(nn.Module):
    pass


