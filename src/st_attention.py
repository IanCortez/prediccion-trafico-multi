"""
Módulos espaciotemporales sobre la semana típica de cada ciudad
[168 horas x C celdas x canales].

UrbanSTGNN (GNNUI, Kaiser et al. 2025)
    Convolución de difusión sobre ventanas de h horas, con la velocidad de
    celdas enmascaradas como tarea auxiliar de interpolación.

TSAEncoder (Crossformer, Zhang & Yan 2023 / iTransformer, Liu et al. 2024)
    La semana de cada celda se parte en segmentos de seg_len horas y se aplica
    atención en dos etapas: temporal dentro de cada celda y espacial entre
    celdas a través de unos pocos routers (O(C·L) en vez de O(C²·L)). Con
    seg_len=168 queda un token por celda, que es el caso de iTransformer.
    No hay embedding posicional por celda para no atar el modelo a la rejilla
    de una ciudad; la identidad de la celda entra por su codificación urbana.

CorridorPace
    Ritmo (s/m) por celda a lo largo de la recta O-D, promediado con pesos
    fijos. La distancia multiplica fuera de la red, así que la predicción
    puede crecer con viajes más largos que los vistos en entrenamiento.
"""

import numpy as np
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
    """Tres capas DGCN como en GNNUI, sobre una ventana de `window` horas aplanada."""

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


def _safe_kpm(kpm):
    # un segmento sin ninguna celda válida daría softmax de todo -inf (NaN)
    return kpm & ~kpm.all(dim=1, keepdim=True)


class TSALayer(nn.Module):
    def __init__(self, d, n_heads=4, n_routers=10, n_segments=28, dropout=0.1, K=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp1 = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.routers = nn.Parameter(torch.randn(n_segments, n_routers, d) * 0.02)
        self.send = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.recv = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n3, self.n4 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp2 = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * d, d))
        # la atención global entre celdas diluye la señal local; la difusión la recupera
        self.local = DiffusionGraphConv(d, d, K)
        self.n5 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z, valid, P_fwd, P_bwd):
        """Z [C, L, d]; valid [C, L] = celda con actividad en el segmento."""
        if Z.shape[1] > 1:
            a, _ = self.time_attn(Z, Z, Z, need_weights=False)
            Z = self.n1(Z + self.drop(a))
            Z = self.n2(Z + self.drop(self.mlp1(Z)))

        Zt = Z.transpose(0, 1)                                  # [L, C, d]
        kpm = _safe_kpm(~valid.transpose(0, 1))
        # los routers solo recogen de celdas activas; luego reparten a todas
        R, _ = self.send(self.routers, Zt, Zt, key_padding_mask=kpm, need_weights=False)
        Zb, _ = self.recv(Zt, R, R, need_weights=False)
        Zt = self.n3(Zt + self.drop(Zb))
        Zt = self.n4(Zt + self.drop(self.mlp2(Zt)))
        Zt = self.n5(Zt + self.drop(F.gelu(self.local(Zt, P_fwd, P_bwd))))
        return Zt.transpose(0, 1)


class TSAEncoder(nn.Module):
    """Codifica la semana completa de una vez y luego se lee (hora, celda) a demanda."""

    def __init__(self, cell_dim, seg_len=6, d=64, out=32, n_layers=2,
                 n_heads=4, n_routers=10, dropout=0.1, K=2):
        super().__init__()
        assert N_HOW % seg_len == 0, "seg_len debe dividir 168 (6, 12, 24, 168...)"
        self.seg_len = seg_len
        self.L = N_HOW // seg_len
        self.out_dim = out
        f_in = F_ST_RAW + 3
        self.embed = nn.Sequential(nn.Linear(seg_len * f_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(self.L, d) * 0.02)
        self.cell_id = nn.Linear(cell_dim, d)
        self.layers = nn.ModuleList([TSALayer(d, n_heads, n_routers, self.L, dropout, K)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.to_hours = nn.Linear(d, seg_len * out)
        self.interp = nn.Linear(out, 1)

    def encode(self, feat, spd, obs, h_cells, P_fwd, P_bwd, node_mask=None):
        """feat [168, C, F_raw], spd/obs [168, C], h_cells [C, cell_dim] -> Z [C, L, d]."""
        Hn, C, _ = feat.shape
        Mk = torch.ones(C, device=feat.device) if node_mask is None else node_mask.float()
        Mk = Mk.view(1, C, 1).expand(Hn, C, 1)
        N = obs.unsqueeze(-1)
        X = torch.cat([feat, spd.unsqueeze(-1) * Mk * N, Mk, N], dim=-1)
        X = X.view(self.L, self.seg_len, C, -1).permute(2, 0, 1, 3).reshape(C, self.L, -1)
        Z = self.embed(X) + self.pos.unsqueeze(0) + self.cell_id(h_cells).unsqueeze(1)
        # actividad por volumen de viajes, que no deriva de la etiqueta
        act = (feat[..., 0] + feat[..., 1]) > 0
        valid = act.view(self.L, self.seg_len, C).any(1).T
        for layer in self.layers:
            Z = layer(Z, valid, P_fwd, P_bwd)
        return self.norm(Z)

    def read(self, Z, taus, cells):
        """Estado en (τ, celda). cells puede ser [B] o [B, K]."""
        seg, off = taus // self.seg_len, taus % self.seg_len
        if cells.dim() == 2:
            seg = seg.unsqueeze(1).expand_as(cells)
            off = off.unsqueeze(1).expand_as(cells)
        h = self.to_hours(Z[cells, seg]).view(*cells.shape, self.seg_len, self.out_dim)
        idx = off[..., None, None].expand(*off.shape, 1, self.out_dim)
        return torch.gather(h, -2, idx).squeeze(-2)

    def reconstruct(self, Z, cells):
        """Velocidad reconstruida de `cells` en las 168 h."""
        h = self.to_hours(Z[cells]).view(len(cells), N_HOW, self.out_dim)
        return self.interp(h).squeeze(-1)


def attach_corridors(cd, city, K=24):
    """Celdas de la recta O-D por par único de celdas, submuestreadas a K como máximo."""
    keys = cd.o_cell.numpy().astype(np.int64) * cd.n_cells + cd.d_cell.numpy()
    uniq, inv = np.unique(keys, return_inverse=True)
    cells = np.zeros((len(uniq), K), dtype=np.int64)
    mask = np.zeros((len(uniq), K), dtype=bool)
    pos = np.zeros((len(uniq), K), dtype=np.float32)
    for p, key in enumerate(uniq):
        lst = city._cells_on_line(int(key // cd.n_cells), int(key % cd.n_cells))
        if len(lst) > K:
            lst = [lst[i] for i in np.linspace(0, len(lst) - 1, K).round().astype(int)]
        n = len(lst)
        cells[p, :n] = lst
        mask[p, :n] = True
        pos[p, :n] = np.linspace(0.0, 1.0, n) if n > 1 else 0.5
    cd.corr_cells = torch.tensor(cells)
    cd.corr_mask = torch.tensor(mask)
    cd.corr_pos = torch.tensor(pos)
    cd.pair_of_trip = torch.tensor(inv.reshape(-1), dtype=torch.long)
    return cd


class CorridorPace(nn.Module):
    """log Σ_k w_k exp(δ_k): log-ritmo del corredor con w_k = fracción de ruta."""

    def __init__(self, cell_dim, st_dim, trip_dim, d=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.cell = nn.Linear(cell_dim, d)
        self.st = nn.Linear(st_dim, d) if st_dim else None
        self.trip = nn.Linear(trip_dim, d)
        self.pos = nn.Sequential(nn.Linear(1, d), nn.Tanh(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.pace = nn.Linear(d, 1)
        nn.init.zeros_(self.pace.weight)        # arranca en el ritmo medio
        nn.init.zeros_(self.pace.bias)

    def forward(self, h_cells, cells, mask, pos, trip_vec, st_states=None):
        x = (self.cell(h_cells[cells]) + self.pos(pos.unsqueeze(-1))
             + self.trip(trip_vec).unsqueeze(1))
        if self.st is not None and st_states is not None:
            x = x + self.st(st_states)
        a, _ = self.attn(x, x, x, key_padding_mask=~mask, need_weights=False)
        x = self.n1(x + a)
        x = self.n2(x + self.ffn(x))
        delta = self.pace(x).squeeze(-1)
        w = mask.float() / mask.float().sum(1, keepdim=True).clamp(min=1.0)
        return torch.logsumexp(delta + torch.log(w.clamp(min=1e-12)), dim=1)
