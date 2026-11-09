"""
Módulos espaciotemporales sobre la semana típica de cada ciudad
[168 horas x C celdas x canales].

UrbanSTGNN
    Convolución de difusión sobre ventanas de h horas, con la velocidad de
    celdas enmascaradas como tarea auxiliar de interpolación.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_HOW = 168          # horas de la semana
F_ST_RAW = 6         # log n_orig, log n_dest, hora sin/cos, día sin/cos


def transition_matrices(src, dst, C):
    """Matrices de transición hacia adelante y hacia atrás, dispersas."""
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)

    def norm(a, b):
        deg = torch.zeros(C).index_add_(0, a, torch.ones(len(a)))
        val = 1.0 / deg[a].clamp(min=1.0)
        return torch.sparse_coo_tensor(torch.stack([a, b]), val, (C, C),
                                       check_invariants=False).coalesce()

    return norm(src, dst), norm(dst, src)


class DiffusionGraphConv(nn.Module):
    """H' = Σ_k T_k(P_f) H Θ_f^k + T_k(P_b) H Θ_b^k, con polinomios de Chebyshev.

    H llega como [B, C, F] y se aplana a [C, B·F] para resolver las B ventanas
    con un solo producto disperso por término.
    """

    def __init__(self, fin, fout, K=2):
        super().__init__()
        self.K = K
        self.theta_f = nn.ModuleList([nn.Linear(fin, fout, bias=False) for _ in range(K)])
        self.theta_b = nn.ModuleList([nn.Linear(fin, fout, bias=False) for _ in range(K)])
        self.bias = nn.Parameter(torch.zeros(fout))

    def _cheb(self, P, X):
        t_prev, t_cur = X, torch.sparse.mm(P, X)
        out = [t_cur]
        for _ in range(2, self.K + 1):
            t_prev, t_cur = t_cur, 2 * torch.sparse.mm(P, t_cur) - t_prev
            out.append(t_cur)
        return out

    def forward(self, H, P_fwd, P_bwd):
        B, C, Fi = H.shape
        X = H.permute(1, 0, 2).reshape(C, B * Fi)
        out = self.bias
        for P, thetas in ((P_fwd, self.theta_f), (P_bwd, self.theta_b)):
            for Tk, th in zip(self._cheb(P, X), thetas):
                out = out + th(Tk.reshape(C, B, Fi).permute(1, 0, 2))
        return out


class UrbanSTGNN(nn.Module):
    """Tres capas DGCN sobre una ventana de `window` horas aplanada."""

    def __init__(self, window=6, hidden=32, out=32, K=2, dropout=0.1):
        super().__init__()
        self.window = window
        fin = window * (F_ST_RAW + 3)          # canales + T + M + N
        self.l1 = DiffusionGraphConv(fin, hidden, K)
        self.l2 = DiffusionGraphConv(hidden, hidden, K)
        self.l3 = DiffusionGraphConv(hidden, out, K)
        self.n1, self.n2, self.n3 = nn.LayerNorm(hidden), nn.LayerNorm(hidden), nn.LayerNorm(out)
        self.drop = nn.Dropout(dropout)
        self.interp = nn.Linear(out, window)
        self.out_dim = out

    def forward(self, H0, P_fwd, P_bwd):
        h1 = self.n1(self.drop(F.relu(self.l1(H0, P_fwd, P_bwd))))
        h2 = self.n2(self.drop(F.relu(self.l2(h1, P_fwd, P_bwd))) + h1)
        return self.n3(self.l3(h2, P_fwd, P_bwd))


def window_index(window):
    """Para cada hora τ de la semana, las horas [τ-h+1, ..., τ] (cíclicas)."""
    return torch.tensor([[(tau - window + 1 + j) % N_HOW for j in range(window)]
                         for tau in range(N_HOW)], dtype=torch.long)


def assemble_st_input(feat, spd, obs, win_idx, taus, node_mask=None):
    """H0 = [X | T·M·N | M | N] para las ventanas que terminan en `taus`.

    M marca celdas ocultas y N celdas-hora sin viajes; así el modelo distingue
    una velocidad nula de un dato oculto o inexistente.
    """
    steps = win_idx[taus]                           # [B, h]
    B = steps.shape[0]
    X = feat[steps]                                 # [B, h, C, F_raw]
    N = obs[steps].unsqueeze(-1)
    M = torch.ones_like(N) if node_mask is None else node_mask.expand_as(N)
    T = spd[steps].unsqueeze(-1) * M * N
    Z = torch.cat([X, T, M, N], dim=-1)
    return Z.permute(0, 2, 1, 3).reshape(B, Z.shape[2], -1)   # [B, C, h·(F_raw+3)]
