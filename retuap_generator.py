"""
Conditional perturbation generators of ReT-UAP (Sec. 4.3 of the manuscript).

Both generators follow a conditional design in which a FIXED noise tensor is
modulated by cross-modal semantic information:

  Image generator G_x (Eqs. 10-12, 14)
      e_txt_i     = norm((1/K) sum_k f_txt(t_{i,k}))           (Eq. 10, primary surrogate)
      delta_hat_i = clip(G_x(z_x; e_txt_i), -eps_x, eps_x)       one candidate per image
      a_i         = softmax_i(f_score(delta_hat_i))              (Eq. 14, one scalar per candidate)
      delta_x     = clip(sum_i a_i delta_hat_i, -eps_x, eps_x)   (Eq. 14)
    z_x is sampled once from N(0, I), is not trained, and is shared by all
    samples, so the candidates differ only through their conditions.

  Text generator G_t (Eq. 13)
      e_img   = norm((1/|B|) sum_{i in B} v_i)                   (Eq. 13, primary surrogate)
      delta_t = G_t(z_t; e_img) in R^{K_p x d_tok}
    with a fixed noise z_t; every row of delta_t is L2-norm-clamped to the range
    of the primary surrogate's token-embedding norms (Sec. 4.3.2).

Upsampling block (Eqs. 11-12). Each block of the deconvolution backbone
(ConvTranspose2d -> GroupNorm -> ReLU layers) is followed by
      z <- z + AdaLN-Zero(z, e),
          AdaLN-Zero(z, e) = g * (alpha * GroupNorm(z) + beta)    (Eq. 11)
      z <- z + Attn(Q(z), K(e), V(e))                            (Eq. 12)
where (g, alpha, beta) come from a zero-initialized linear projection of the
condition e and e enters the cross-attention as a single token. As in the
standard (DiT) AdaLN-Zero design, the modulation is added through a residual
connection and alpha is parameterized as 1 + alpha', so that the zero-initialized
projection (g = alpha' = beta = 0) makes every block start as the identity while
the gate g still receives gradients. The cross-attention output projection is
zero-initialized for the same reason.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Conditioning modules (Eqs. 11-12)
# ==========================================

class AdaLNZero(nn.Module):
    """Eq. (11): g * (alpha * GroupNorm(z) + beta), with (g, alpha', beta) from a
    zero-initialized linear projection of the condition and alpha = 1 + alpha'."""

    def __init__(self, channels, context_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, channels), channels, affine=False)
        self.proj = nn.Linear(context_dim, 3 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z, e):
        # z: (N, C, H, W), e: (N, context_dim)
        gate, alpha, beta = self.proj(e)[:, :, None, None].chunk(3, dim=1)
        return gate * ((1.0 + alpha) * self.norm(z) + beta)


class CrossAttention(nn.Module):
    """Eq. (12): Attn(Q(z), K(e), V(e)) with the spatial positions of z as
    queries and the condition e as a single key/value token."""

    def __init__(self, channels, context_dim, num_heads=1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, z, e):
        n, c, h, w = z.shape
        q = self.to_q(z.flatten(2).transpose(1, 2))                    # (N, HW, C)
        k = self.to_k(e).unsqueeze(1)                                  # (N, 1, C)
        v = self.to_v(e).unsqueeze(1)                                  # (N, 1, C)
        q = q.view(n, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(n, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(n, 1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.head_dim ** -0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(n, h * w, c)
        return self.to_out(out).transpose(1, 2).reshape(n, c, h, w)


# ==========================================
# Conditional generator backbone
# ==========================================

class ConditionalGenerator(nn.Module):
    """Deconvolution generator driven by a fixed noise tensor and a condition.

    z_shape     : shape of the fixed noise (without batch dimension), e.g. (100, 3, 3)
    num_filters : channels of the upsampling blocks, e.g. [[1024, 512], [256, 128], [64, 32]]
    The first layer uses `first_kernel_size` with stride 1; every further layer
    (and the output layer) doubles the spatial size.
    forward(e) returns tanh outputs of shape (N, out_channels, H, W), one per
    condition row; the fixed noise is shared across the N rows.
    """

    def __init__(self, z_shape, num_filters, out_channels, first_kernel_size,
                 context_dim, num_heads=1, num_groups=8):
        super().__init__()
        # Fixed (non-learnable) noise, sampled once from N(0, I) (Sec. 4.3.1).
        self.register_buffer('z', torch.randn(1, *z_shape))

        self.backbone = nn.ModuleList()
        self.adaln = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        in_ch = z_shape[0]
        for i, filters in enumerate(num_filters):
            layers = []
            for j, out_ch in enumerate(filters):
                if i == 0 and j == 0:
                    deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=first_kernel_size,
                                                stride=1, padding=0)
                else:
                    deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
                nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                nn.init.constant_(deconv.bias, 0.0)
                layers += [deconv, nn.GroupNorm(min(num_groups, out_ch), out_ch), nn.ReLU()]
                in_ch = out_ch
            self.backbone.append(nn.Sequential(*layers))
            self.adaln.append(AdaLNZero(in_ch, context_dim, num_groups))
            self.cross_attn.append(CrossAttention(in_ch, context_dim, num_heads))

        out_conv = nn.ConvTranspose2d(in_ch, out_channels, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(out_conv.weight, mean=0.0, std=0.02)
        nn.init.constant_(out_conv.bias, 0.0)
        self.output_layer = nn.Sequential(out_conv, nn.Tanh())

    def forward(self, e):
        z = self.z.expand(e.shape[0], *self.z.shape[1:])
        for block, adaln, attn in zip(self.backbone, self.adaln, self.cross_attn):
            z = block(z)
            z = z + adaln(z, e)   # Eq. (11), residual AdaLN-Zero
            z = z + attn(z, e)    # Eq. (12)
        return self.output_layer(z)


# ==========================================
# Scoring network f_score (Eq. 14)
# ==========================================

class ScoreNet(nn.Module):
    """f_score: maps ONE candidate perturbation to ONE scalar score.

    Small strided convolutions on the candidate (scaled by 1/eps_x so that the
    input is O(1)), global average pooling and a two-layer MLP whose last layer
    is zero-initialized, so that a_i = softmax_i(f_score) is uniform at the start
    of training (Sec. 4.3.2).
    """

    def __init__(self, in_channels=3, width=16, eps=1.0):
        super().__init__()
        self.eps = eps
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, width), width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, 2 * width), 2 * width),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.ReLU(),
                                  nn.Linear(2 * width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, delta_hat):
        # delta_hat: (N, C, H, W) -> (N,)
        return self.head(self.features(delta_hat / self.eps)).squeeze(-1)


# ==========================================
# Image and text perturbation generators
# ==========================================

class ImagePerturbationGenerator(nn.Module):
    """G_x plus the scoring network f_score (Eqs. 10-12, 14).

    context_dim : feature dimension of the primary surrogate (condition e_txt)
    eps         : l_inf budget eps_x (in [0, 1] pixel units)
    image_size  : output resolution; G_x natively produces 384 x 384 and is
                  bilinearly resized if the data resolution differs.
    """

    def __init__(self, context_dim, eps, image_size=384, z_dim=100, num_heads=1):
        super().__init__()
        self.eps = eps
        self.image_size = image_size
        self.G_x = ConditionalGenerator(
            z_shape=(z_dim, 3, 3), num_filters=[[1024, 512], [256, 128], [64, 32]],
            out_channels=3, first_kernel_size=4, context_dim=context_dim, num_heads=num_heads)
        self.f_score = ScoreNet(in_channels=3, eps=eps)

    def forward(self, e_txt):
        """e_txt: (B, context_dim) per-image conditions (Eq. 10).
        Returns (delta_x (1, 3, H, W), delta_hat (B, 3, H, W), a (B,))."""
        out = self.G_x(e_txt)
        if out.shape[-1] != self.image_size or out.shape[-2] != self.image_size:
            out = F.interpolate(out, size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=False)
        delta_hat = torch.clamp(out, -self.eps, self.eps)
        a = torch.softmax(self.f_score(delta_hat), dim=0)
        delta_x = (a.view(-1, 1, 1, 1) * delta_hat).sum(dim=0, keepdim=True)
        delta_x = torch.clamp(delta_x, -self.eps, self.eps)
        return delta_x, delta_hat, a


class TextPerturbationGenerator(nn.Module):
    """G_t (Eq. 13): one universal token-embedding perturbation per call.

    context_dim   : feature dimension of the primary surrogate (condition e_img)
    num_positions : K_p, the number of substituted token positions
    token_dim     : d_tok, the primary surrogate's token-embedding dimension
                    (768 for the BERT-tokenizer models, 512 for CLIP); must be a
                    multiple of 256 (the backbone outputs 16 x (token_dim / 16)).
    min_norm, max_norm : L2-norm range of the primary's token-embedding table.
    """

    def __init__(self, context_dim, num_positions, token_dim, min_norm, max_norm,
                 z_dim=32, num_heads=1):
        super().__init__()
        assert token_dim % 256 == 0, 'token_dim must be a multiple of 256'
        self.num_positions = num_positions
        self.token_dim = token_dim
        self.G_t = ConditionalGenerator(
            z_shape=(z_dim, 1, token_dim // 256), num_filters=[[256, 128], [64, 32]],
            out_channels=num_positions, first_kernel_size=1, context_dim=context_dim,
            num_heads=num_heads)
        self.register_buffer('min_norm', torch.tensor(float(min_norm)))
        self.register_buffer('max_norm', torch.tensor(float(max_norm)))

    def forward(self, e_img):
        """e_img: (1, context_dim) batch-level condition (Eq. 13).
        Returns delta_t of shape (K_p, token_dim) with norm-clamped rows."""
        raw = self.G_t(e_img).reshape(self.num_positions, self.token_dim)
        norm = raw.norm(dim=1, keepdim=True)
        return raw * (norm.clamp(self.min_norm, self.max_norm) / (norm + 1e-8))
