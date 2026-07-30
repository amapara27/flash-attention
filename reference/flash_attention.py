# per block reference of flash attention 2

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

# single head 
# different d_k and d_v vals to ensure correct tran
# small vals to read traces
N = 8
d_k = 4
d_v = 6
B_c = 3 
scale = d_k**-0.5

# matrix initialization
Q = torch.randn([N, d_k])
K = torch.randn([N, d_k])
V = torch.randn([N, d_v])

def flash_attention(Q, K, V):
    # online softmax vals
    m = torch.full((N, 1), float('-inf'))
    l = torch.full((N, 1), 0)

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

        l = corr * l + P_j.sum(-1, keepdim=True)

        O_acc = corr * O_acc + P_j @ V_j # correct prev O and add new

        m = m_new

    O = O_acc / l # final normalization
    return O

# check
print(torch.allclose(flash_attention(Q, K, V),  F.scaled_dot_product_attention(Q, K, V), atol=1e-5))