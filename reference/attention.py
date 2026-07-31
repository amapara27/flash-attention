import torch
import math

def attention(Q, K, V):
  d_k = Q.shape[-1]
  scale = 1 / math.sqrt(d_k)
  S = Q @ K.mT * scale
  P = torch.softmax(S, dim = -1)
  O = P @ V
  return O