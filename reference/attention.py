import torch
import math

def attention(Q, K, V, causal=False):
  N = Q.shape[-2]
  d_k = Q.shape[-1]

  scale = 1 / math.sqrt(d_k)
  S = Q @ K.transpose(-1, -2) * scale

  if causal: 
    mask = torch.ones(N, N, dtype=torch.bool, device = Q.device).triu(1)
    S = S.masked_fill(mask, float('-inf'))

  P = torch.softmax(S, dim = -1)
  O = P @ V
  return O