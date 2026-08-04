import torch
import math

def attention(Q, K, V, causal=False):
  N = Q.shape[0]
  d_k = Q.shape[-1]

  scale = 1 / math.sqrt(d_k)
  S = Q @ K.mT * scale

  if causal: 
    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
    S = S.masked_fill(mask, float('-inf'))

  P = torch.softmax(S, dim = -1)
  O = P @ V
  return O