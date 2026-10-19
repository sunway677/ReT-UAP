"""
Surrogate wrappers, the margin contrastive loss and the UCB surrogate selection
of ReT-UAP (Secs. 4.1, 4.3.2, 4.4 and 4.5 of the manuscript).

Surrogate pool (Sec. 5.1, Sec. 4.5.1)
  Six VLP models: ALBEF, TCL, BLIP, XVLM (BERT-tokenizer fused encoders) and
  CLIP ViT-B/16, ViT-B/32. One of them is the primary surrogate, the other five
  are auxiliaries. Auxiliaries of the primary's family are same-system (their
  token embeddings accept delta_t directly), the others are cross-system.

Surrogate (uniform interface over the six models)
  clean features        v_i = norm(f_img(x_i)),  u_i = norm(f_txt(t_i))           (Eq. 1)
  adversarial features  v_adv_i = norm(f_img(x_i + delta_x)),
                        u_adv_i = norm(f_txt(t_{i,1} (+) delta_t))               (Eq. 15)
  (+) replaces the token embeddings of the K_p most critical words of t_{i,1} by
  the rows of delta_t. Critical words are ranked by the KL divergence between
  the softmax of the clean text feature and that of the text feature with the
  word masked (Sec. 4.3.2), computed with the surrogate that is being attacked.
  BERT-family models mask with '[UNK]' and count positions in word pieces; CLIP
  models mask with the content-free special token <|startoftext|> (CLIP has no
  [UNK]/[MASK] token, and <|endoftext|> marks the pooled position) and count
  positions in CLIP BPE tokens. A position is an (offset, n_extra) pair: offset
  is the index of the word's first token counted without [CLS] / <|startoftext|>
  and n_extra the number of further tokens of the same word.

margin_contrastive_loss (Eqs. 16-22)
  l_i(p) = -log( sim_m(a_i, p_i) / (sim_m(a_i, p_i) + Omega_i) ),
  sim_m(a, p) = exp((a . p - m_i) / tau)                                 (Eqs. 16, 19)
  Omega_i = sum_{j in B} exp(a_i . c_j / tau) + sum_{n in N_B} exp(a_i . n / tau)  (Eq. 21)
  L = 1/(2|B|) sum_i (l_i(p_a) + l_i(p_b))                               (Eqs. 17, 22)

UCBSurrogateSelector (Eqs. 24-27)
  L_hat(s) = (L(s) - mu_s) / (sigma_s + eps0)                           (Eq. 24)
  R(s) <- (1 - eta) R(s) + eta L_hat(s)                                 (Eq. 25)
  UCB(s) = R(s) + c_ucb sqrt(log(T + 1) / (n_s + eps0))                 (Eq. 26)
  gate   = lambda_sur * sigmoid(R(s*))                                  (Eq. 27)
  plus a per-epoch minimum selection floor floor(n_bat * rho_min).
"""

import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


BERT_FAMILY = ('ALBEF', 'TCL', 'BLIP', 'XVLM')
CLIP_FAMILY = ('ViT-B/16', 'ViT-B/32')
SURROGATE_POOL = BERT_FAMILY + CLIP_FAMILY

MAX_TEXT_LEN = 30            # BERT-family caption length in tokens (incl. [CLS]), as in C-PGC
CLIP_MASK_TOKEN = '<|startoftext|>'

# Image normalization applied to [0, 1] pixels before every surrogate.
IMAGE_NORMALIZATION = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                           (0.26862954, 0.26130258, 0.27577711))


def model_tag(name):
    """File-system tag of a model name ('ViT-B/16' -> 'ViT-B_16')."""
    return name.replace('/', '_')


def model_family(name):
    if name in BERT_FAMILY:
        return 'bert'
    if name in CLIP_FAMILY:
        return 'clip'
    raise ValueError(f'unknown surrogate {name!r}; expected one of {SURROGATE_POOL}')


# ==========================================
# Surrogate wrapper
# ==========================================

class Surrogate:
    """Frozen VLP surrogate with the encoders needed by ReT-UAP.

    model     : a vendored retrieval model (models.model_retrieval.ALBEF for ALBEF/TCL,
                BLIP_Retrieval, XVLM, or a CLIP model from models.clip_model)
    tokenizer : the BERT tokenizer of a BERT-family model (unused for CLIP)
    Image inputs are [0, 1] pixel tensors at the data resolution; CLIP models
    resize them (and the perturbation) to their input resolution.
    """

    def __init__(self, name, model, tokenizer=None):
        self.name = name
        self.tag = model_tag(name)
        self.family = model_family(name)
        self.is_clip = self.family == 'clip'
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device
        if self.is_clip:
            self.image_size = int(self.model.visual.input_resolution)
            self.context_length = int(self.model.context_length)
            from models.clip_model import clip as clip_module
            self._clip = clip_module
            self.mask_token_id = clip_module._tokenizer.encoder[CLIP_MASK_TOKEN]
        else:
            self.image_size = None

    def autocast(self):
        if self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ---------------- token embeddings ----------------
    def token_embedding_weight(self):
        if self.is_clip:
            return self.model.token_embedding.weight
        encoder = self.model.text_encoder
        encoder = getattr(encoder, 'bert', encoder)
        return encoder.embeddings.word_embeddings.weight

    @property
    def token_dim(self):
        return int(self.token_embedding_weight().shape[1])

    # ---------------- images ----------------
    def _resize(self, x):
        if self.is_clip and (x.shape[-1] != self.image_size or x.shape[-2] != self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)
        return x

    def image_features(self, images, delta_x=None):
        """norm(f_img(clip(x + delta_x, 0, 1))) (Eq. 1 without, Eq. 15 with delta_x).
        images: (B, 3, H, W) in [0, 1]; delta_x: (1, 3, H', W') or None."""
        x = self._resize(images)
        if delta_x is not None:
            d = self._resize(delta_x)
            if d.shape[-2:] != x.shape[-2:]:
                d = F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.clamp(x + d, 0.0, 1.0)
        x = IMAGE_NORMALIZATION(x)
        with self.autocast():
            if self.is_clip:
                feat = self.model.encode_image(x)
            else:
                feat = self.model.inference_image(x)['image_feat']
        return F.normalize(feat.float(), dim=-1)

    # ---------------- texts ----------------
    def _bert_inputs(self, texts):
        return self.tokenizer(list(texts), padding='max_length', truncation=True,
                              max_length=MAX_TEXT_LEN, return_tensors='pt').to(self.device)

    def _clip_encode_ids(self, tokens):
        with self.autocast():
            return self.model.encode_text(tokens.to(self.device)).float()

    def raw_text_features(self, texts):
        """f_txt(t) before L2 normalization: the projected [CLS] feature of the
        BERT-family wrappers (text_proj of the first token) and the projected
        <|endoftext|> feature of CLIP. Eq. (10) averages these raw features over the
        K captions and then normalizes; Eq. (1) normalizes a single one."""
        if self.is_clip:
            return self._clip_encode_ids(self._clip.tokenize(list(texts), self.context_length, True))
        with self.autocast():
            out = self.model.inference_text(self._bert_inputs(texts))
            return self.model.text_proj(out['text_embed'][:, 0, :]).float()

    def text_features(self, texts):
        """u = norm(f_txt(t)) (Eq. 1)."""
        return F.normalize(self.raw_text_features(texts), dim=-1)

    def adv_text_features(self, texts, delta_t, positions, extras):
        """u_adv = norm(f_txt(t (+) delta_t)) (Eq. 15): row j of delta_t replaces the
        token embedding of the j-th position (positions[i][j], extras[i][j])."""
        with self.autocast():
            if self.is_clip:
                out = self.model.inference_text_replace(list(texts), delta_t, positions, extras)
            else:
                out = self.model.inference_text_replace(self._bert_inputs(texts), delta_t,
                                                        positions, extras)
        return F.normalize(out['text_feat'].float(), dim=-1)

    # ---------------- critical token positions ----------------
    def _word_units(self, text):
        """Per space-separated word: token ids (CLIP) or word pieces (BERT)."""
        words = text.split(' ')
        if self.is_clip:
            return words, self._clip.encode_words(text)
        return words, [self.tokenizer.tokenize(w) for w in words]

    def _max_offset(self):
        # Last token index that can hold a substitution: index 1 + offset must lie
        # before the 30-token limit (BERT, no trailing [SEP] with the vendored
        # tokenizer) or before <|endoftext|> in the 77-token context (CLIP).
        if self.is_clip:
            return self.context_length - 3
        return MAX_TEXT_LEN - 2

    @torch.no_grad()
    def critical_positions(self, texts, num_positions, batch_size=256):
        """Top-num_positions critical words of every text (Sec. 4.3.2).

        Returns (positions, extras): lists (one per text) of the (offset, n_extra)
        of the selected words, most critical first. Words whose first token falls
        outside the encoded caption, or that have no token, are not selectable.
        """
        texts = list(texts)
        clean = self.text_features(texts)
        units = [self._word_units(t) for t in texts]
        masked, owner = [], []
        for i, (words, pieces) in enumerate(units):
            for w in range(len(words)):
                if self.is_clip:
                    ids = [tok for k, p in enumerate(pieces) for tok in (p if k != w else [self.mask_token_id])]
                    masked.append(ids)
                else:
                    masked.append(' '.join(words[:w] + ['[UNK]'] + words[w + 1:]))
                owner.append(i)
        feats = []
        for s in range(0, len(masked), batch_size):
            chunk = masked[s:s + batch_size]
            if self.is_clip:
                f = self._clip_encode_ids(self._clip.pack_tokens(chunk, self.context_length))
            else:
                f = self.raw_text_features(chunk)
            feats.append(F.normalize(f, dim=-1))
        feats = torch.cat(feats, dim=0) if feats else clean[:0]
        owner = torch.as_tensor(owner, dtype=torch.long, device=clean.device)
        p_clean = clean.softmax(dim=-1)[owner]
        kl = F.kl_div(feats.log_softmax(dim=-1), p_clean, reduction='none').sum(dim=-1).cpu().tolist()

        positions, extras = [], []
        max_offset = self._max_offset()
        k = 0
        for words, pieces in units:
            scores = kl[k:k + len(words)]
            k += len(words)
            cand, offset = [], 0
            for w, p in enumerate(pieces):
                if len(p) > 0 and offset <= max_offset:
                    cand.append((scores[w], -w, offset, len(p) - 1))
                offset += len(p)
            cand.sort(reverse=True)  # highest KL first; ties keep the earlier word
            sel = cand[:num_positions]
            positions.append([c[2] for c in sel])
            extras.append([c[3] for c in sel])
        return positions, extras


# ==========================================
# Margin contrastive loss (Eqs. 16-22)
# ==========================================

def margin_contrastive_loss(adv, targets, margin, batch_clean, negatives=None, temperature=0.07):
    """Margin-aware dual-path contrastive loss.

    adv         : (B, d) normalized adversarial features (v_adv or u_adv)
    targets     : pair of (B, d) soft targets (Eq. 9 and its text-branch analogue)
    margin      : (B,) adaptive margins m_i (Eq. 18)
    batch_clean : (B, d) clean features of the batch in the adversarial
                  feature's modality, including the sample's own one (Eq. 21)
    negatives   : (K_neg, d) informed negatives of the same modality, or None
    The margin enters sim(.) of the target term in the numerator and in the
    denominator (Eq. 19 substituted into Eq. 16); the same Omega_i is used for
    both targets. Returns 1/(2|B|) sum_i (l_i(p_a) + l_i(p_b)).
    """
    neg_logits = adv @ batch_clean.t() / temperature
    if negatives is not None and negatives.shape[0] > 0:
        neg_logits = torch.cat([neg_logits, adv @ negatives.t() / temperature], dim=1)
    log_omega = torch.logsumexp(neg_logits, dim=1)
    total = adv.new_zeros(adv.shape[0])
    for target in targets:
        pos = ((adv * target).sum(dim=-1) - margin) / temperature
        total = total + torch.logaddexp(pos, log_omega) - pos
    return total.sum() / (len(targets) * adv.shape[0])


# ==========================================
# UCB surrogate selection (Eqs. 24-27)
# ==========================================

class UCBSurrogateSelector:
    """Selects one auxiliary surrogate per step after warm-up (Algorithm 1).

    names            : auxiliary surrogate names
    c_ucb            : exploration coefficient of Eq. (26)
    eta              : reward momentum of Eq. (25)
    lambda_sur       : auxiliary loss scale of Eq. (27)
    min_select_ratio : rho_min of the selection floor
    norm_ema_rate    : update rate r of the running loss statistics of Eq. (24):
                       exponential moving averages of L and L^2 with Adam-style
                       bias correction (divided by 1 - (1 - r)^c after c
                       observations), mu = E[L], sigma^2 = max(E[L^2] - mu^2, 0).
                       The first observation therefore gives L_hat = 0, and the
                       statistics are unbiased while the averages warm up. The
                       manuscript gives neither the rate nor the initialization.
    eps0             : the small constant of Eqs. (24) and (26)

    Per step: s = select(); compute L(s); L_hat = update(s, L(s)) (Eqs. 24-25);
    w = gate(s) (Eq. 27, uses the updated reward).

    Selection floor: at the start of every post-warm-up epoch, start_epoch(n_bat)
    sets floor = floor(n_bat * rho_min) (lowered to floor(n_bat / S_aux) if the
    floors of all auxiliaries cannot fit into the epoch). select() follows the
    UCB score, except when the number of remaining steps of the epoch equals the
    total number of selections still missing to reach every auxiliary's floor;
    from then on it picks, by UCB score, among the auxiliaries below the floor.
    Every auxiliary is therefore selected at least `floor` times per epoch, and
    UCB is overridden only when that is necessary.
    """

    def __init__(self, names, c_ucb=1.0, eta=0.1, lambda_sur=0.65, min_select_ratio=0.05,
                 norm_ema_rate=0.01, eps0=1e-6):
        self.names = list(names)
        self.k = len(self.names)
        self.c_ucb = c_ucb
        self.eta = eta
        self.lambda_sur = lambda_sur
        self.min_select_ratio = min_select_ratio
        self.norm_ema_rate = norm_ema_rate
        self.eps0 = eps0
        self.R = np.zeros(self.k)                   # running rewards R_bar(s)
        self.n = np.zeros(self.k, dtype=np.int64)   # selections n_s
        self.T = 0                                  # total selections
        self.m1 = np.zeros(self.k)                  # EMA of L   (before bias correction)
        self.m2 = np.zeros(self.k)                  # EMA of L^2 (before bias correction)
        self.count = np.zeros(self.k, dtype=np.int64)
        self.mu = np.zeros(self.k)                  # bias-corrected mean of L
        self.var = np.zeros(self.k)                 # bias-corrected variance of L
        self.floor = 0
        self.epoch_counts = np.zeros(self.k, dtype=np.int64)
        self.steps_left = 0

    def start_epoch(self, n_batches):
        floor = int(math.floor(n_batches * self.min_select_ratio))
        if self.k > 0 and floor * self.k > n_batches:
            print(f'[UCB] floor {floor} x {self.k} auxiliaries exceeds {n_batches} steps; '
                  f'using {n_batches // self.k}')
            floor = n_batches // self.k
        self.floor = floor
        self.epoch_counts = np.zeros(self.k, dtype=np.int64)
        self.steps_left = n_batches

    def ucb_scores(self):
        return self.R + self.c_ucb * np.sqrt(np.log(self.T + 1) / (self.n + self.eps0))

    def select(self):
        scores = self.ucb_scores()
        deficit = np.maximum(self.floor - self.epoch_counts, 0)
        if deficit.sum() > 0 and deficit.sum() >= self.steps_left:
            candidates = np.flatnonzero(deficit > 0)
        else:
            candidates = np.arange(self.k)
        s = int(candidates[np.argmax(scores[candidates])])
        self.epoch_counts[s] += 1
        self.steps_left -= 1
        self.n[s] += 1
        self.T += 1
        return s

    def update(self, s, loss):
        loss = float(loss)
        r = self.norm_ema_rate
        self.m1[s] = (1.0 - r) * self.m1[s] + r * loss
        self.m2[s] = (1.0 - r) * self.m2[s] + r * loss * loss
        self.count[s] += 1
        corr = 1.0 - (1.0 - r) ** int(self.count[s])
        self.mu[s] = self.m1[s] / corr
        self.var[s] = max(self.m2[s] / corr - self.mu[s] ** 2, 0.0)
        l_hat = (loss - self.mu[s]) / (math.sqrt(self.var[s]) + self.eps0)       # Eq. (24)
        self.R[s] = (1.0 - self.eta) * self.R[s] + self.eta * l_hat               # Eq. (25)
        return l_hat

    def gate(self, s):
        return self.lambda_sur / (1.0 + math.exp(-float(self.R[s])))              # Eq. (27)

    def summary(self):
        lines = ['UCB surrogate selector:']
        for i, name in enumerate(self.names):
            lines.append(f'  {name:9s} R={self.R[i]:+.4f} n={int(self.n[i]):6d} '
                         f'epoch={int(self.epoch_counts[i]):4d} mu={self.mu[i]:.4f} '
                         f'sigma={math.sqrt(self.var[i]):.4f} gate={self.gate(i):.4f}')
        return '\n'.join(lines)

    def state_dict(self):
        return {'names': list(self.names), 'R': self.R.tolist(), 'n': self.n.tolist(), 'T': int(self.T),
                'm1': self.m1.tolist(), 'm2': self.m2.tolist(), 'count': self.count.tolist(),
                'mu': self.mu.tolist(), 'var': self.var.tolist()}
