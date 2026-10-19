"""Offline adversarial target construction of ReT-UAP (Sec. 4.2 and Sec. 4.4).

For one surrogate model, this module turns the clean feature bank of the
training set into the quantities that stay fixed during generator training:
paired cross-modal prototypes, OT-based transport weights, soft adversarial
targets, adaptive margins, and the transport distributions used to pick
informed negatives. Every surrogate in the pool owns one
:class:`SurrogateTargets` instance built on its own embedding space.

Mapping to the manuscript (equation numbers as in the manuscript)
------------------------------------------------------------------
* Eq. (1)  Clean feature bank v_i, u_i. Computed by the caller (first caption
  t_{i,1} of every training image) and passed to ``SurrogateTargets``, which
  only re-applies the L2 normalization.
* Eq. (2)  Cross-modal clustering and paired prototypes:
  :func:`kmeans_labels`, :func:`drop_empty_clusters`, :func:`paired_prototypes`.
* Eq. (3)  Sample-level hubness h(.): :func:`hubness_counts`.
* Eq. (4)  Cluster-level hubness with min-max normalization:
  :func:`cluster_hubness`.
* Eqs. (5)-(6)  Cross-modal diversity (assignment histogram and normalized
  entropy, then min-max over clusters): :func:`cross_modal_diversity`.
* Sec. 4.2.5  Semantic mismatch rank r_{i,c}: :func:`mismatch_rank`.
* Eq. (7)  OT score psi = lambda_r r + lambda_h h + lambda_d d, cost M = -psi:
  :func:`ot_score`.
* Eq. (8)  Entropic OT with uniform marginals (Sinkhorn, log domain) and the
  row-normalized transport weights W: :func:`sinkhorn_log`.
* Eq. (9)  Soft adversarial targets: :func:`soft_targets`.
* Eq. (18) Adaptive margins from the entropy of W_i: :func:`adaptive_margins`.
* Eq. (20) and Sec. 4.4 "Informed Negatives" steps (1)-(3):
  :func:`select_informed_negatives`. The selected bank indices give the
  vectors n of Eq. (21), which the loss (in the trainer) consumes.

Branches
--------
Every quantity exists twice, once per branch (cross-modal targeting principle,
Sec. 4.2.2). In the code a branch is described by its *anchor* bank A (the
bank that is clustered) and its *paired* bank B (the other modality, whose
features are perturbed in that branch):

==========  ==================  =====================  ======================
branch      anchor A (cluster)  paired B (perturbed)   prototypes (P_A, P_B)
==========  ==================  =====================  ======================
``'img'``   U (text bank)       V (image bank)         (p^(u), p^(v))
``'txt'``   V (image bank)      U (text bank)          (q^(v), q^(u))
==========  ==================  =====================  ======================

With this notation both branches use the same code:

* hubness (Eq. 3) of every anchor element a_j counts the queries b_i whose
  top-K_h anchor neighbours (similarity a_l . b_i over all N anchor elements)
  contain a_j;
* diversity (Eq. 5) assigns each member j of an anchor cluster to
  argmax_r' b_j . P_B[r'];
* the mismatch rank of sample i ranks P_A by descending b_i . P_A[c];
* the soft targets (Eq. 9) are norm(W P_A) and norm(W P_B).

Soft targets returned by :meth:`SurrogateTargets.targets` follow Eq. (9) as
printed, in the order of the manuscript symbols:

* ``branch='img'`` -> (v~tar(u), u~tar(u)) with
  v~tar(u)_i = norm(sum_c W^(u)_ic p^(u)_c)  (built from the TEXT-space
  prototypes p^(u)) and u~tar(u)_i = norm(sum_c W^(u)_ic p^(v)_c)  (built
  from the paired IMAGE-space prototypes p^(v));
* ``branch='txt'`` -> (v~tar(v), u~tar(v)) with
  v~tar(v)_i = norm(sum_c W^(v)_ic q^(v)_c)  (IMAGE-space prototypes q^(v))
  and u~tar(v)_i = norm(sum_c W^(v)_ic q^(u)_c)  (paired TEXT-space
  prototypes q^(u)).

In other words, the first target of each branch is a mixture of the
prototypes of the clustered (opposite) modality and the second one a mixture
of their paired prototypes in the perturbed modality. The loss (Eqs. 17 and
22) uses both targets with the same denominator, so the order of the pair does
not change the computation.

Details the manuscript leaves open, and the choices made here
-------------------------------------------------------------
* K-Means: ``sklearn.cluster.MiniBatchKMeans`` on the L2-normalized anchor
  bank with ``random_state=seed``, ``batch_size=min(4096, N)``, ``n_init=3``,
  ``max_iter=300``. Only the cluster labels are used; prototypes are the
  normalized member sums of Eq. (2), not the K-Means centres.
* Empty clusters (a K-Means centre that is nobody's nearest centre) have no
  prototype in Eq. (2). They are dropped before scoring and OT, so every
  quantity uses the number of non-empty clusters C' <= C (marginal 1/C',
  rank normalization by C' - 1, entropy normalization by log C').
* Min-max normalization of a constant vector returns zeros (Eqs. 4, 6, 18).
* K_h larger than N is clipped to N.
* Sinkhorn (Eq. 8) runs exactly ``ot_iters`` iterations in the log domain
  (scaled dual potentials, logsumexp) in ``ot_dtype`` (float64 by default),
  starting from zero potentials and alternating row and column updates; the
  last update is the column update. The kernel never underflows.
* Informed negatives (Sec. 4.4): the top-1 prototype of bank sample j is
  argmax_c W_jc. Candidates are the bank samples outside the batch whose top-1
  prototype differs from the top-1 prototype of every batch sample. If fewer
  than K_neg candidates exist, all of them are returned.
* K_neg is not given in the manuscript. The default ``DEFAULT_K_NEG = 21`` is
  floor(64 / 3), the value the research code used for the image loss with
  the batch size of 64 (it took floor(|B| / 3) negatives per step); the same
  value is used for both branches here because both losses now run on the
  |B| = 64 first captions/images of the batch.

Computational notes
-------------------
Hubness is computed with chunked top-K over query rows, so the N x N
similarity matrix is never materialized. The rank, OT score and Sinkhorn steps
work on N x C' matrices (about 45 M entries for Flickr30K / MS-COCO with
C = 5% of N); Sinkhorn is chunked over rows. The transport weights W are kept
(float32) because informed-negative selection needs every row of W at each
training step.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

__all__ = [
    'DEFAULT_K_NEG', 'BRANCHES', 'SurrogateTargets', 'num_clusters',
    'minmax_normalize', 'kmeans_labels', 'drop_empty_clusters',
    'paired_prototypes', 'hubness_counts', 'cluster_hubness',
    'cross_modal_diversity', 'mismatch_rank', 'ot_score', 'sinkhorn_log',
    'soft_targets', 'adaptive_margins', 'select_informed_negatives',
]

# Number of informed negatives per branch and step (not specified in the
# manuscript; floor(64 / 3) as in the research code, see module docstring).
DEFAULT_K_NEG = 21

BRANCHES = ('img', 'txt')

_STATE_VERSION = 1

# Budget (number of matrix entries) for one chunk of a chunked N x M operation.
_CHUNK_ENTRIES = 1 << 24


def _row_chunk(n_rows, n_cols, chunk_size=None):
    """Return the number of rows per chunk for an (n_rows, n_cols) operation."""
    if chunk_size is not None:
        return max(1, int(chunk_size))
    return max(1, min(n_rows, _CHUNK_ENTRIES // max(1, n_cols)))


def num_clusters(n_samples, cluster_ratio):
    """Number of prototypes C = round(cluster_ratio * N) (Sec. 5.1), at least 2."""
    return max(2, int(math.floor(cluster_ratio * n_samples + 0.5)))


def minmax_normalize(x):
    """Min-max normalization to [0, 1]; a constant vector maps to zeros."""
    lo, hi = x.min(), x.max()
    if float(hi - lo) <= 0.0:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


# ----------------------------------------------------------------------------
# Eq. (2): clustering and paired prototypes
# ----------------------------------------------------------------------------

def kmeans_labels(feats, n_clusters, seed=42):
    """MiniBatch K-Means cluster labels of an L2-normalized bank (Eq. 2).

    Args:
        feats: (N, d) tensor, the anchor bank of the branch.
        n_clusters: C.
        seed: random state of K-Means.

    Returns:
        (N,) int64 numpy array with labels in {0, ..., C-1}.
    """
    x = F.normalize(feats.detach().float(), dim=-1).cpu().numpy()
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         batch_size=min(4096, x.shape[0]), n_init=3,
                         max_iter=300)
    return km.fit_predict(x).astype(np.int64)


def drop_empty_clusters(labels, n_clusters):
    """Remove clusters without members and relabel to 0..C'-1.

    Returns:
        (new_labels, kept): new_labels is an (N,) int64 numpy array, kept the
        sorted original ids of the C' non-empty clusters.
    """
    labels = np.asarray(labels, dtype=np.int64)
    sizes = np.bincount(labels, minlength=n_clusters)
    kept = np.flatnonzero(sizes > 0)
    remap = np.full(n_clusters, -1, dtype=np.int64)
    remap[kept] = np.arange(kept.size)
    return remap[labels], kept


def paired_prototypes(anchor, paired, labels, n_clusters):
    """Paired cross-modal prototypes of Eq. (2).

    P_A[c] = norm(sum_{i in C_c} a_i) and P_B[c] = norm(sum_{i in C_c} b_i),
    both summed over the same member set C_c of the anchor clustering. For the
    image branch (anchor U, paired V) these are p^(u) and p^(v); for the text
    branch (anchor V, paired U) they are q^(v) and q^(u).

    Args:
        anchor: (N, d_a) anchor bank (clustered modality).
        paired: (N, d_b) paired bank (other modality).
        labels: (N,) labels in {0, ..., n_clusters-1}, every cluster non-empty.
        n_clusters: number of clusters.

    Returns:
        (P_A, P_B): (C, d_a) and (C, d_b) L2-normalized float32 tensors on the
        device of ``anchor``.
    """
    # Summed on the CPU in float64 so the result does not depend on the
    # (non-deterministic) order of atomic additions on a GPU.
    lab = torch.as_tensor(labels, dtype=torch.long).cpu()
    protos = []
    for bank in (anchor, paired):
        acc = torch.zeros(n_clusters, bank.shape[1], dtype=torch.float64)
        acc.index_add_(0, lab, bank.detach().cpu().double())
        protos.append(F.normalize(acc, dim=-1).float().to(anchor.device))
    return protos[0], protos[1]


# ----------------------------------------------------------------------------
# Eqs. (3)-(4): hubness
# ----------------------------------------------------------------------------

@torch.no_grad()
def hubness_counts(queries, gallery, k, chunk_size=None):
    """Sample-level hubness of Eq. (3).

    h(g_j) = sum_i 1[g_j in TopK(q_i)], where TopK(q_i) are the k gallery
    elements with the largest cosine similarity g_l . q_i over all l = 1..N.
    Image branch: queries V, gallery U (h(u_j)); text branch: queries U,
    gallery V (h(v_j)). Computed in chunks of query rows.

    Returns:
        (N_gallery,) int64 tensor of counts (they sum to N_queries * k).
    """
    n_q, n_g = queries.shape[0], gallery.shape[0]
    k = min(int(k), n_g)
    counts = torch.zeros(n_g, dtype=torch.long, device=gallery.device)
    step = _row_chunk(n_q, n_g, chunk_size)
    g_t = gallery.float().t()
    for s in range(0, n_q, step):
        sim = queries[s:s + step].float() @ g_t
        top = sim.topk(k, dim=1).indices
        counts += torch.bincount(top.reshape(-1), minlength=n_g)
    return counts


def cluster_hubness(counts, labels, n_clusters):
    """Cluster-level hubness of Eq. (4): member mean of h, min-max over clusters."""
    lab = torch.as_tensor(labels, dtype=torch.long, device=counts.device)
    total = torch.zeros(n_clusters, dtype=torch.float64, device=counts.device)
    total.index_add_(0, lab, counts.double())
    size = torch.bincount(lab, minlength=n_clusters).double()
    return minmax_normalize(total / size).float()


# ----------------------------------------------------------------------------
# Eqs. (5)-(6): cross-modal diversity
# ----------------------------------------------------------------------------

@torch.no_grad()
def cross_modal_diversity(paired, proto_paired, labels, n_clusters,
                          chunk_size=None):
    """Cross-modal diversity of Eqs. (5)-(6), min-max over clusters.

    For every member j of anchor cluster c, r_j = argmax_r' b_j . P_B[r']
    (image branch: v_j against the paired image prototypes p^(v)); pi_{c,r} is
    the fraction of members with r_j = r and
    d_c = -(1 / log C) sum_r pi_{c,r} log pi_{c,r} (0 log 0 = 0).

    Returns:
        (C,) float32 tensor in [0, 1].
    """
    if n_clusters < 2:
        raise ValueError('cross_modal_diversity needs at least 2 clusters')
    n = paired.shape[0]
    step = _row_chunk(n, n_clusters, chunk_size)
    p_t = proto_paired.float().t()
    nearest = torch.cat([(paired[s:s + step].float() @ p_t).argmax(dim=1)
                         for s in range(0, n, step)])
    lab = torch.as_tensor(labels, dtype=torch.long, device=paired.device)
    hist = torch.zeros(n_clusters * n_clusters, dtype=torch.float64,
                       device=paired.device)
    hist.index_add_(0, lab * n_clusters + nearest,
                    torch.ones(n, dtype=torch.float64, device=paired.device))
    hist = hist.view(n_clusters, n_clusters)
    pi = hist / hist.sum(dim=1, keepdim=True)
    ent = -torch.special.xlogy(pi, pi).sum(dim=1) / math.log(n_clusters)
    return minmax_normalize(ent).float()


# ----------------------------------------------------------------------------
# Sec. 4.2.5, Eqs. (7)-(9): mismatch rank, OT score, Sinkhorn, soft targets
# ----------------------------------------------------------------------------

@torch.no_grad()
def mismatch_rank(queries, proto_anchor, chunk_size=None):
    """Semantic mismatch rank r_{i,c} in [0, 1] (Sec. 4.2.5).

    r_{i,c} = (position of prototype c when the C prototypes are sorted by
    descending similarity b_i . P_A[c]) / (C - 1); 0 = most similar,
    1 = least similar. Image branch: v_i against p^(u); text branch: u_i
    against q^(v). Ties keep the prototype order (stable sort).

    Returns:
        (N, C) float32 tensor.
    """
    n, c = queries.shape[0], proto_anchor.shape[0]
    if c < 2:
        raise ValueError('mismatch_rank needs at least 2 prototypes')
    out = torch.empty(n, c, dtype=torch.float32, device=queries.device)
    step = _row_chunk(n, c, chunk_size)
    p_t = proto_anchor.float().t()
    for s in range(0, n, step):
        sim = queries[s:s + step].float() @ p_t
        order = torch.argsort(sim, dim=1, descending=True, stable=True)
        pos = torch.arange(c, dtype=torch.float32, device=queries.device)
        ranks = torch.empty_like(sim)
        ranks.scatter_(1, order, pos.expand_as(sim).contiguous())
        out[s:s + step] = ranks / (c - 1)
    return out


def ot_score(rank, hub, div, weights=(0.8, 0.1, 0.1), dtype=torch.float64):
    """OT score of Eq. (7): psi_{i,c} = l_r r_{i,c} + l_h h_c + l_d d_c.

    The transport cost of Eq. (8) is M = -psi.

    Returns:
        (N, C) tensor of dtype ``dtype``.
    """
    l_r, l_h, l_d = (float(w) for w in weights)
    psi = rank.to(dtype) * l_r
    psi += (l_h * hub.to(dtype) + l_d * div.to(dtype)).unsqueeze(0)
    return psi


@torch.no_grad()
def sinkhorn_log(cost, reg=0.005, n_iters=500, chunk_size=None,
                 overwrite_cost=False):
    """Entropic OT of Eq. (8) by log-domain Sinkhorn, and the weights W.

    Solves min_Pi <Pi, M> + reg * sum Pi (log Pi - 1) subject to
    Pi 1_C = 1/N 1_N and Pi^T 1_N = 1/C 1_C. With scaled dual potentials
    f, g the plan is Pi = exp(-M / reg + f 1^T + 1 g^T); the iterations are

        f_i = log(1/N) - logsumexp_c(-M_ic / reg + g_c)
        g_c = log(1/C) - logsumexp_i(-M_ic / reg + f_i)

    starting from g = 0, for exactly ``n_iters`` rounds. W is Pi
    row-normalized, i.e. W_ic = softmax_c(-M_ic / reg + g_c).

    Args:
        cost: (N, C) cost matrix M (its dtype is the working precision).
        reg: entropic regularization eps_ot.
        n_iters: number of Sinkhorn iterations.
        chunk_size: rows per chunk (auto if None).
        overwrite_cost: reuse the memory of ``cost`` for the log-kernel.

    Returns:
        (W, stats): W is an (N, C) float32 tensor whose rows sum to 1; stats
        holds the maximum relative marginal errors of Pi
        (``row_err`` = max_i |N (Pi 1)_i - 1|, ``col_err`` = max_c
        |C (Pi^T 1)_c - 1|), the corresponding absolute errors
        (``row_abs_err`` = max_i |(Pi 1)_i - 1/N|, ``col_abs_err``) and the
        relative column error of the row-normalized W
        (``w_col_err`` = max_c |(C / N) sum_i W_ic - 1|). The column errors
        are at machine precision because the last update is the column update.
    """
    n, c = cost.shape
    log_k = cost.neg_() if overwrite_cost else cost.neg()
    log_k.div_(reg)
    dt, dev = log_k.dtype, log_k.device
    log_a, log_b = -math.log(n), -math.log(c)
    step = _row_chunk(n, c, chunk_size)
    spans = [(s, min(s + step, n)) for s in range(0, n, step)]
    f = torch.zeros(n, dtype=dt, device=dev)
    g = torch.zeros(c, dtype=dt, device=dev)

    def col_lse(f_vec):
        parts = torch.stack([torch.logsumexp(log_k[s:e] + f_vec[s:e, None], dim=0)
                             for s, e in spans])
        return torch.logsumexp(parts, dim=0)

    for _ in range(int(n_iters)):
        for s, e in spans:
            f[s:e] = log_a - torch.logsumexp(log_k[s:e] + g, dim=1)
        g = log_b - col_lse(f)

    w = torch.empty(n, c, dtype=torch.float32, device=dev)
    row_lse = torch.empty(n, dtype=dt, device=dev)
    for s, e in spans:
        z = log_k[s:e] + g
        lse = torch.logsumexp(z, dim=1)
        row_lse[s:e] = lse
        w[s:e] = torch.exp(z - lse[:, None]).float()
    row_mass = torch.exp(f + row_lse)
    col_mass = torch.exp(g + col_lse(f))
    w_col = w.double().sum(dim=0) * (c / n)
    stats = {
        'row_err': float((row_mass * n - 1).abs().max()),
        'col_err': float((col_mass * c - 1).abs().max()),
        'row_abs_err': float((row_mass - 1.0 / n).abs().max()),
        'col_abs_err': float((col_mass - 1.0 / c).abs().max()),
        'w_col_err': float((w_col - 1).abs().max()),
    }
    return w, stats


def soft_targets(w, proto_anchor, proto_paired):
    """Soft adversarial targets of Eq. (9).

    Returns (norm(W P_A), norm(W P_B)). Image branch: (v~tar(u), u~tar(u)) =
    (norm(W^(u) p^(u)), norm(W^(u) p^(v))). Text branch: (v~tar(v), u~tar(v)) =
    (norm(W^(v) q^(v)), norm(W^(v) q^(u))).
    """
    return (F.normalize(w @ proto_anchor.float(), dim=-1),
            F.normalize(w @ proto_paired.float(), dim=-1))


def adaptive_margins(w, margin_min=0.01, margin_max=0.05):
    """Adaptive margins of Eq. (18).

    H_i = -sum_c W_ic log W_ic (0 log 0 = 0), H^_i = min-max of H over the N
    bank samples, m_i = m_max - H^_i (m_max - m_min).

    Returns:
        (entropy, margins): two (N,) float32 tensors.
    """
    ent = -torch.special.xlogy(w.double(), w.double()).sum(dim=1)
    h_hat = minmax_normalize(ent)
    m = margin_max - h_hat * (margin_max - margin_min)
    return ent.float(), m.float()


# ----------------------------------------------------------------------------
# Eq. (20): informed negatives
# ----------------------------------------------------------------------------

@torch.no_grad()
def select_informed_negatives(w, top1, idx, k_neg):
    """Informed negatives of Sec. 4.4 (steps (1)-(3), Eq. 20).

    (1) Candidates: bank samples j outside the batch with
        top1(j) != top1(i) for every batch sample i (top1 = argmax_c W).
    (2) Score: w_j . w_bar_B, w_bar_B = mean of the batch rows of W (Eq. 20).
    (3) The K_neg candidates with the highest score, in descending order.

    Args:
        w: (N, C) transport weights of the branch.
        top1: (N,) argmax_c W.
        idx: (B,) long tensor of batch indices into the bank.
        k_neg: number of negatives; fewer are returned if fewer candidates
            exist.

    Returns:
        (K,) long tensor of bank indices, K = min(k_neg, #candidates).
    """
    if k_neg <= 0:
        return torch.empty(0, dtype=torch.long, device=w.device)
    valid = ~torch.isin(top1, top1[idx])
    valid[idx] = False
    n_valid = int(valid.sum())
    if n_valid == 0:
        return torch.empty(0, dtype=torch.long, device=w.device)
    score = w @ w[idx].mean(dim=0)
    score = score.masked_fill(~valid, float('-inf'))
    return score.topk(min(int(k_neg), n_valid)).indices


# ----------------------------------------------------------------------------
# Per-surrogate container
# ----------------------------------------------------------------------------

class SurrogateTargets:
    """Offline OT targets, margins and informed negatives of one surrogate.

    Args:
        img_feats: V, (N, d) clean image features of the training set
            (Eq. 1), one row per training image, for this surrogate.
        txt_feats: U, (N, d) clean features of the first caption of each
            training image (Eq. 1), same row order as ``img_feats``.
        n_clusters: C (Sec. 5.1: round(cluster_ratio * N), see
            :func:`num_clusters`).
        hub_k: K_h of Eq. (3).
        score_weights: (lambda_r, lambda_h, lambda_d) of Eq. (7).
        ot_reg: eps_ot of Eq. (8).
        ot_iters: number of Sinkhorn iterations.
        margin_range: (m_min, m_max) of Eq. (18).
        seed: K-Means random state.
        device: device for all tensors and computations (K-Means runs in
            scikit-learn on the CPU).
        ot_dtype: working precision of the Sinkhorn solver.
        chunk_size: rows per chunk for chunked operations (auto if None).
        verbose: print a short summary per branch.

    Usage::

        st = SurrogateTargets(V, U, num_clusters(len(V), 0.05), device='cuda')
        st.build()                         # or st.load_state_dict(cached)
        v_tar, u_tar = st.targets(idx, 'img')
        m = st.margins(idx, 'img')
        neg = st.informed_negatives(idx, 'img', DEFAULT_K_NEG)  # use V[neg]
    """

    def __init__(self, img_feats, txt_feats, n_clusters, hub_k=20,
                 score_weights=(0.8, 0.1, 0.1), ot_reg=0.005, ot_iters=500,
                 margin_range=(0.01, 0.05), seed=42, device='cpu',
                 ot_dtype=torch.float64, chunk_size=None, verbose=True):
        self.device = torch.device(device)
        img = torch.as_tensor(img_feats).detach().float().to(self.device)
        txt = torch.as_tensor(txt_feats).detach().float().to(self.device)
        if img.dim() != 2 or txt.dim() != 2 or img.shape[0] != txt.shape[0]:
            raise ValueError('img_feats and txt_feats must be (N, d) with the same N, '
                             f'got {tuple(img.shape)} and {tuple(txt.shape)}')
        self.V = F.normalize(img, dim=-1)
        self.U = F.normalize(txt, dim=-1)
        self.N = self.V.shape[0]
        n_clusters = int(n_clusters)
        if not 2 <= n_clusters <= self.N:
            raise ValueError(f'n_clusters must be in [2, N={self.N}], got {n_clusters}')
        weights = tuple(float(x) for x in score_weights)
        if len(weights) != 3 or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError('score_weights must be three numbers summing to 1 (Eq. 7), '
                             f'got {score_weights}')
        m_min, m_max = (float(x) for x in margin_range)
        if m_min > m_max:
            raise ValueError(f'margin_range must satisfy m_min <= m_max, got {margin_range}')
        self.config = {
            'n_clusters': n_clusters, 'hub_k': int(hub_k),
            'score_weights': weights, 'ot_reg': float(ot_reg),
            'ot_iters': int(ot_iters), 'margin_range': (m_min, m_max),
            'seed': int(seed), 'ot_dtype': str(ot_dtype).replace('torch.', ''),
        }
        self.ot_dtype = ot_dtype
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._branches = {}

    # ------------------------------------------------------------------ build
    def _banks(self, branch):
        """(anchor, paired) banks of a branch: 'img' -> (U, V), 'txt' -> (V, U)."""
        if branch == 'img':
            return self.U, self.V
        if branch == 'txt':
            return self.V, self.U
        raise ValueError(f"branch must be 'img' or 'txt', got {branch!r}")

    @torch.no_grad()
    def _build_branch(self, branch):
        cfg = self.config
        anchor, paired = self._banks(branch)
        labels_raw = kmeans_labels(anchor, cfg['n_clusters'], cfg['seed'])
        labels, kept = drop_empty_clusters(labels_raw, cfg['n_clusters'])
        c_eff = int(kept.size)
        if c_eff < 2:
            raise RuntimeError(f'[{branch}] fewer than 2 non-empty clusters')
        proto_a, proto_b = paired_prototypes(anchor, paired, labels, c_eff)
        counts = hubness_counts(paired, anchor, cfg['hub_k'], self.chunk_size)
        hub = cluster_hubness(counts, labels, c_eff)
        div = cross_modal_diversity(paired, proto_b, labels, c_eff, self.chunk_size)
        rank = mismatch_rank(paired, proto_a, self.chunk_size)
        psi = ot_score(rank, hub, div, cfg['score_weights'], self.ot_dtype)
        del rank
        w, stats = sinkhorn_log(psi.neg_(), cfg['ot_reg'], cfg['ot_iters'],
                                self.chunk_size, overwrite_cost=True)
        del psi
        tgt_a, tgt_b = soft_targets(w, proto_a, proto_b)
        ent, margins = adaptive_margins(w, *cfg['margin_range'])
        top1 = w.argmax(dim=1)
        sizes = np.bincount(labels, minlength=c_eff)
        stats.update({
            'n_clusters_kept': c_eff,
            'n_clusters_empty': cfg['n_clusters'] - c_eff,
            'cluster_size_min': int(sizes.min()),
            'cluster_size_max': int(sizes.max()),
            'top1_unique': int(torch.unique(top1).numel()),
            'margin_mean': float(margins.mean()),
        })
        out = {
            'labels': torch.as_tensor(labels, dtype=torch.long, device=self.device),
            'kept_clusters': torch.as_tensor(kept, dtype=torch.long),
            'proto_anchor': proto_a, 'proto_paired': proto_b,
            'hub_counts': counts, 'hub': hub, 'div': div,
            'W': w, 'top1': top1, 'entropy': ent, 'margins': margins,
            'target_anchor': tgt_a, 'target_paired': tgt_b,
            'stats': stats,
        }
        if self.verbose:
            print(f'  [Targets:{branch}] N={self.N} C={cfg["n_clusters"]} '
                  f'kept={c_eff} (empty {stats["n_clusters_empty"]}) '
                  f'size[min={stats["cluster_size_min"]}, max={stats["cluster_size_max"]}] | '
                  f'Sinkhorn row_err={stats["row_err"]:.2e} col_err={stats["col_err"]:.2e} '
                  f'W col_err={stats["w_col_err"]:.2e} | top1 unique={stats["top1_unique"]} | '
                  f'margin mean={stats["margin_mean"]:.4f}')
        return out

    def build(self):
        """Compute both branches (Eqs. 2-9 and 18) and cache them in memory."""
        for branch in BRANCHES:
            self._branches[branch] = self._build_branch(branch)
        return self

    @property
    def is_built(self):
        return all(b in self._branches for b in BRANCHES)

    def _branch(self, branch):
        if branch not in BRANCHES:
            raise ValueError(f"branch must be 'img' or 'txt', got {branch!r}")
        if branch not in self._branches:
            raise RuntimeError('SurrogateTargets: call build() or load_state_dict() first')
        return self._branches[branch]

    def _index(self, idx):
        return torch.as_tensor(idx, dtype=torch.long).reshape(-1).to(self.device)

    # ------------------------------------------------------------- accessors
    def targets(self, idx, branch):
        """Soft targets of Eq. (9) for bank indices ``idx``.

        ``branch='img'`` -> (v~tar(u), u~tar(u)) = (norm(W^(u) p^(u)),
        norm(W^(u) p^(v))), targets of the adversarial image features.
        ``branch='txt'`` -> (v~tar(v), u~tar(v)) = (norm(W^(v) q^(v)),
        norm(W^(v) q^(u))), targets of the adversarial text features.
        Each is a (B, d) L2-normalized tensor.
        """
        b = self._branch(branch)
        i = self._index(idx)
        return b['target_anchor'][i], b['target_paired'][i]

    def margins(self, idx, branch):
        """(B,) adaptive margins m_i of Eq. (18) for the branch."""
        return self._branch(branch)['margins'][self._index(idx)]

    def informed_negatives(self, idx, branch, k_neg=DEFAULT_K_NEG):
        """Bank indices of the informed negatives N_B of the batch ``idx``.

        Implements Sec. 4.4 "Informed Negatives" (steps 1-3, Eq. 20). The
        caller uses V[n] (image branch) or U[n] (text branch) as the vectors n
        of Eq. (21); :meth:`negative_features` returns them directly.
        """
        b = self._branch(branch)
        return select_informed_negatives(b['W'], b['top1'], self._index(idx), k_neg)

    def negative_features(self, idx, branch, k_neg=DEFAULT_K_NEG):
        """Clean bank features of the informed negatives: V[n] ('img') or U[n] ('txt')."""
        n = self.informed_negatives(idx, branch, k_neg)
        return (self.V if branch == 'img' else self.U)[n]

    def transport_weights(self, branch):
        """(N, C') row-normalized transport weights W of the branch (Eq. 8)."""
        return self._branch(branch)['W']

    def stats(self, branch):
        """Summary statistics of the branch (cluster sizes, marginal errors, ...)."""
        return dict(self._branch(branch)['stats'])

    # ---------------------------------------------------------- persistence
    def _fingerprint(self):
        n = torch.arange(1, self.N + 1, dtype=torch.float64, device=self.device)
        vals = []
        for x in (self.V, self.U):
            x = x.double()
            vals += [x.sum(), x.abs().sum(), (x.sum(dim=1) * n).sum() / self.N]
        return {'shape_img': tuple(self.V.shape), 'shape_txt': tuple(self.U.shape),
                'sums': torch.stack(vals).cpu()}

    def state_dict(self):
        """Serializable state (config, input fingerprint, both branches) on the CPU."""
        if not self.is_built:
            raise RuntimeError('SurrogateTargets: nothing to save, call build() first')
        branches = {}
        for name, b in self._branches.items():
            branches[name] = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in b.items()}
        return {'version': _STATE_VERSION, 'config': dict(self.config),
                'fingerprint': self._fingerprint(), 'branches': branches}

    def load_state_dict(self, d):
        """Restore a state produced by :meth:`state_dict`.

        Raises ValueError if the cached state was built with a different
        configuration or from different feature banks, so a stale cache is
        never reused silently.
        """
        if d.get('version') != _STATE_VERSION:
            raise ValueError(f'unsupported SurrogateTargets state version {d.get("version")}')
        cfg = dict(d['config'])
        cfg['score_weights'] = tuple(cfg['score_weights'])
        cfg['margin_range'] = tuple(cfg['margin_range'])
        if cfg != self.config:
            raise ValueError(f'cached targets config {cfg} differs from {self.config}')
        fp_now, fp_old = self._fingerprint(), d['fingerprint']
        if (tuple(fp_old['shape_img']) != fp_now['shape_img']
                or tuple(fp_old['shape_txt']) != fp_now['shape_txt']
                or not torch.allclose(fp_old['sums'], fp_now['sums'],
                                      rtol=1e-5, atol=1e-5)):
            raise ValueError('cached targets were built from different feature banks')
        self._branches = {}
        for name in BRANCHES:
            b = d['branches'][name]
            self._branches[name] = {
                k: (v.to(self.device) if torch.is_tensor(v) and k != 'kept_clusters' else v)
                for k, v in b.items()}
        return self
