# per block reference of flash attention 2

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

from attention import attention

def flash_attention(Q, K, V, B_c, trace=False):
    trace_log = []

    N, d_k = Q.shape
    d_v = V.shape[-1]
    scale = d_k**-0.5

    # online softmax vals
    m = torch.full((N, 1), float('-inf'))
    ell = torch.zeros((N, 1))

    # output matrix
    O_acc = torch.zeros((N, d_v))

    # dividing our K, V matrices
    T_c = math.ceil(N / B_c )

    # iterate over K, V
    for j in range(T_c):
        start = j * B_c # start of K, V tile
        end = min(start + B_c, N) # end of K, V tile - can't go past N

        K_j = K[start:end, ]
        V_j = V[start:end, ]

        assert K_j.shape[1] == d_k and V_j.shape[1] == d_v

        S_j = Q @ K_j.T * scale # [N, B_c]

        m_j = S_j.amax(-1, keepdim=True) # local rowmax - [N, 1]
        m_new = torch.maximum(m, m_j) # global rowmax

        corr = torch.exp(m - m_new) # sum correction 
        P_j = torch.exp(S_j - m_new) # [N, B_c]

        row_sum = P_j.sum(-1, keepdim=True)

        ell = corr * ell + row_sum

        O_acc = corr * O_acc + P_j @ V_j # correct prev O and add new

        m = m_new

        if trace:
            trace_log.append({
                "m":      m.clone(),
                "ell":      ell.clone(),
                "O_acc":  O_acc.clone(),
                "corr":   corr.clone(),
                "rowsum": row_sum.clone()
            })

    O = O_acc / ell # final normalization
    return (O, trace_log) if trace else O