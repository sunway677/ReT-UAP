# ReT-UAP

PyTorch implementation of **"ReT-UAP: Rethinking Multi-Surrogate Universal Adversarial
Perturbations on Vision–Language Pre-trained Models"**.

> **Status:** the manuscript is currently **under review at *Information Fusion***.

Ziyu Wang, Jian Li, Yiming Du, Rui Ning, Jiang Li, Daniel Takabi, Lusi Li

ReT-UAP learns a single pair of universal adversarial perturbations (an image
perturbation `delta_x` and a text perturbation `delta_t`) against image–text retrieval
of vision–language pre-trained (VLP) models, using a pool of heterogeneous surrogates
(fused-encoder ALBEF, TCL, BLIP, X-VLM and aligned-encoder CLIP ViT-B/16, ViT-B/32):

- **Targeted structural guidance.** Each surrogate clusters its clean feature bank into
  paired cross-modal prototypes and assigns every sample soft bimodal targets by entropic
  optimal transport, scored by semantic mismatch with hubness and cross-modal diversity
  as structural priors.
- **Conditional universal-perturbation generators.** Modality-specific generators with
  AdaLN-Zero and cross-attention conditioning, trained with a margin-aware dual-path
  contrastive objective with informed negatives.
- **Adaptive surrogate selection.** After a warm-up, an Upper Confidence Bound (UCB)
  policy with normalized rewards activates one auxiliary surrogate per step, so the
  per-step cost stays close to single-surrogate training.

The code builds on the [C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks)
codebase (*"One Perturbation is Enough"*, Fang et al., ICCV 2025) for the VLP model
wrappers, data loading and evaluation protocol.

This repository implements the method as described in the manuscript. Results vary
between training runs and may differ from the numbers reported in the manuscript.

## Repository layout

```
ReT-UAP/
├── retuap_targets.py      # Offline target construction (Sec. 4.2): prototypes (Eq. 2),
│                            hubness (Eqs. 3-4), diversity (Eqs. 5-6), OT assignment
│                            (Eqs. 7-9), adaptive margins (Eq. 18), informed negatives (Eq. 20)
├── retuap_generator.py    # Conditional generators G_x, G_t (AdaLN-Zero + cross-attention,
│                            Eqs. 11-13) and the candidate scoring network (Eq. 14)
├── retuap_attacker.py     # Surrogate wrappers, perturbation construction (Eqs. 14-15),
│                            margin contrastive loss (Eqs. 16-22), UCB selector (Eqs. 24-27)
├── retuap_train.py        # Training (Algorithm 1, Eq. 28)
├── eval.py                # Evaluation: ASR at R@1/5/10 for TR (I2T) and IR (T2I)
├── attacker.py            # Word substitution used to realize delta_t at test time (C-PGC)
├── utils.py, dataset.py   # Model loading and image-text datasets (C-PGC)
├── models/                # VLP backbones: ALBEF/TCL, BLIP, X-VLM, CLIP
├── configs/               # Flickr30K and MS-COCO train/test configs
├── data_annotation/       # Flickr30K and MS-COCO train/test annotation JSONs
├── word_idx_BERT.npy, word_idx_CLIP.npy   # Whole-word vocabulary filters
├── scripts/
│   ├── generate_word_embeddings.py        # Builds the token-embedding tables
│   ├── train.sh                           # bash scripts/train.sh <PRIMARY> <DATASET>
│   └── eval.sh                            # bash scripts/eval.sh  <PRIMARY> <DATASET>
├── logs/                  # SLURM stdout/stderr
├── LICENSE                # MIT (ReT-UAP code)
├── THIRD_PARTY_NOTICES.md # Licenses of the vendored third-party code and data
└── licenses/
```

## Notation: manuscript symbols to command-line flags

All defaults of `retuap_train.py` follow Sec. 5.1 of the manuscript.

| Manuscript | Flag | Default |
|---|---|---|
| epochs `E`, warm-up `E_warm` | `--epochs`, `--aux_warmup_epochs` | `100`, `40` |
| batch size, learning rate (Adam) | `--batch_size`, `--lr` | `64`, `2e-4` |
| image budget `eps_x` | `--eps` (in 1/255) | `12` |
| perturbed token positions `K_p` | `--num_text_positions` (alias `--word_num`) | `1` |
| temperature `tau` (both branches) | `--temperature` | `0.07` |
| clusters `C` (fraction of the dataset size) | `--cluster_ratio` | `0.05` |
| hubness neighbourhood `K_h` | `--hub_k` | `20` |
| OT score weights `(lambda_r, lambda_h, lambda_d)` | `--score_weights` | `0.8 0.1 0.1` |
| entropic regularization `eps_ot`, Sinkhorn iterations | `--ot_reg`, `--ot_iters` | `0.005`, `500` |
| margin range `[m_min, m_max]` | `--margin_min`, `--margin_max` | `0.01`, `0.05` |
| informed negatives `K_neg` | `--k_neg` | `21` |
| UCB exploration `c_ucb`, reward momentum `eta` | `--ucb_exploration`, `--reward_momentum` | `1.0`, `0.1` |
| auxiliary loss scale `lambda_sur`, selection floor `rho_min` | `--lambda_sur`, `--min_select_ratio` | `0.65`, `0.05` |

## Setup

```bash
conda env create -f environment.yml
conda activate retuap
```

`environment.yml` installs `requirements.txt` (Python 3.8). transformers >= 5 is not
supported by the vendored ALBEF/BLIP/X-VLM text encoders. Training and evaluation run on
a CUDA GPU; `bert-base-uncased` (tokenizer/config) is downloaded from the HuggingFace Hub
and the CLIP weights to `~/.cache/clip` on first use.

### Datasets and checkpoints

1. **Flickr30K**: download the [images](https://shannon.cs.illinois.edu/DenotationGraph/)
   and set `image_root` in `configs/Retrieval_flickr_{train,test}.yaml` to the directory
   that **contains** `flickr30k-images/` (default `''` = `./flickr30k-images/`).
   **MS-COCO**: download the [2014 validation images](https://cocodataset.org/#download)
   and set `image_root` in `configs/Retrieval_coco_{train,test}.yaml` to the directory
   that contains `val2014/` (default `./data/coco`). Following C-PGC, the MS-COCO
   training (30,000 images) and test (5,000 images) splits are both drawn from val2014.
   The annotation files are included under `data_annotation/`.
2. Download the retrieval checkpoints of [ALBEF](https://github.com/salesforce/ALBEF),
   [TCL](https://github.com/uta-smile/TCL), [BLIP](https://github.com/salesforce/BLIP)
   (ViT-B) and [X-VLM](https://github.com/zengyan-97/X-VLM) fine-tuned on each dataset,
   and save them as:
   ```
   checkpoint/
   ├── ALBEF/{flickr30k,mscoco}.pth
   ├── TCL/{flickr30k,mscoco}.pth
   ├── BLIP/{flickr30k,mscoco}.pth
   └── XVLM/{flickr30k,mscoco}.pth
   ```
   CLIP ViT-B/16 and ViT-B/32 use the OpenAI weights.
3. Build the token-embedding tables of the six models:
   ```bash
   python scripts/generate_word_embeddings.py --all
   ```
   This writes `word_embedding_<TAG>_<DATASET>.pth` (TAG = model name with `/` replaced by
   `_`). Training uses the primary surrogate's table to norm-clamp `delta_t`; evaluation
   uses the source model's table to map `delta_t` to a vocabulary word.

## Training

```bash
bash scripts/train.sh ALBEF flickr30k        # or: sbatch scripts/train.sh "ViT-B/16" mscoco
```

`<PRIMARY>` is one of `ALBEF`, `TCL`, `BLIP`, `XVLM`, `ViT-B/16`, `ViT-B/32`; the other
five models form the auxiliary pool. Same-system auxiliaries (same text-embedding family
as the primary) supervise both generators; cross-system auxiliaries supervise only the
image generator. Extra `retuap_train.py` flags can be appended to the command. For SLURM,
uncomment and set the `#SBATCH -p` partition line in the scripts and submit from the
repository root.

**Offline stage** (once per surrogate, cached in `output_retuap/cache/<DATASET>/`): clean
feature banks from the first caption of every training image (Eq. 1), MiniBatch K-Means
prototypes, hubness and diversity scores, entropic OT target assignment, soft targets,
adaptive margins, and the critical token positions used to insert `delta_t`.

**Online stage** (Algorithm 1), for every mini-batch:
1. the image generator produces one candidate per sample, conditioned on the primary
   surrogate's mean caption embedding of that image (Eq. 10); the candidates are clipped,
   aggregated with the learned scoring weights (Eq. 14) and clipped to `eps_x`;
2. the text generator produces `delta_t` from the batch-level image prototype (Eq. 13),
   norm-clamped to the L2 range of natural word embeddings, and inserted at the critical
   position(s) of the first caption;
3. the primary surrogate gives `L_img` + `L_txt` (Eqs. 16-22); after the warm-up, the UCB
   policy selects one auxiliary surrogate whose gated loss is added (Eqs. 23-27);
4. one backward pass and one Adam step on the total loss (Eq. 28).

**Outputs** in `output_retuap/<TAG>/<DATASET>/`: `uap_noise-<epoch>.pth` (`delta_x*`,
shape `(1, 3, 384, 384)`), `uap_embedding-<epoch>.pth` (`delta_t*`, shape `(K_p, d_tok)`)
and `generators-<epoch>.pth`, i.e. the perturbations produced at the final training
iteration (only the latest epoch is kept).

## Evaluation

```bash
bash scripts/eval.sh ALBEF flickr30k
```

The perturbations of one primary surrogate are evaluated on all six models on the
Flickr30K 1K or MS-COCO 5K test split. Following C-PGC, every test image receives
`delta_x*` and every test caption receives the adversarial word obtained from `delta_t*`
(nearest whole word of the source model's token-embedding table, substituted at the
caption's most important word); the adversarial captions are built once and shared by
all target models. TR (image-to-text) uses adversarial image queries against the
adversarial caption gallery, and IR (text-to-image) adversarial caption queries against
the adversarial image gallery.

The attack success rate of a target model is

`ASR@K = |C_K ∩ F_K| / |C_K|`,

where `C_K` is the set of queries retrieved correctly at rank `K` without the attack and
`F_K` the set of queries that fail at rank `K` under the attack. `C_K` is computed for each
target model in the same query order as the adversarial pass. The script prints the
results and writes `outputs/result_retuap_<TAG>_<DATASET>/<TAG>_<DATASET>.xlsx` (ASR@1/5/10
for TR and IR; the manuscript reports ASR@1).

## Implementation notes

Settings the manuscript does not fix, chosen here:

- training images that also appear in the test split are removed from the training set
  (the shipped Flickr30K training annotation contains 955 of the 1,000 test images);
- mini-batches are reshuffled every epoch and the last incomplete batch is dropped;
- empty K-Means clusters are dropped before scoring and transport; Sinkhorn runs in the
  log domain;
- `K_neg = 21`; the running loss statistics of Eq. (24) are bias-corrected exponential
  moving averages with rate `--norm_ema_rate 0.01`; Adam betas are `(0.5, 0.999)`;
- the norm range of `delta_t` is taken over the whole-word tokens of the primary's
  token-embedding table (`word_idx_BERT.npy` / `word_idx_CLIP.npy`);
- critical positions for inserting `delta_t` during training are the words with the
  largest KL divergence; the test-time word substitution of C-PGC additionally skips
  stop words;
- surrogate forward passes use bfloat16 autocast on GPU, and CLIP surrogates receive the
  image and `delta_x` resized to their input resolution.

## Citation

If you find this code useful, please cite:

```bibtex
@unpublished{wang2026retuap,
  title  = {{ReT-UAP}: Rethinking Multi-Surrogate Universal Adversarial Perturbations on Vision--Language Pre-trained Models},
  author = {Wang, Ziyu and Li, Jian and Du, Yiming and Ning, Rui and Li, Jiang and Takabi, Daniel and Li, Lusi},
  note   = {Manuscript under review at Information Fusion},
  year   = {2026}
}
```

## Acknowledgements

The model wrappers, data loading and evaluation protocol are adapted from
[C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks):

```bibtex
@inproceedings{fang2025one,
  title     = {One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models},
  author    = {Fang, Hao and Kong, Jiawei and Yu, Wenbo and Chen, Bin and Li, Jiawei and Wu, Hao and Xia, Shu-Tao and Xu, Ke},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4090--4100},
  year      = {2025},
  doi       = {10.1109/ICCV51701.2025.00390}
}
```

We also thank the authors of [ALBEF](https://github.com/salesforce/ALBEF),
[TCL](https://github.com/uta-smile/TCL), [BLIP](https://github.com/salesforce/BLIP),
[X-VLM](https://github.com/zengyan-97/X-VLM) and [CLIP](https://github.com/openai/CLIP)
for releasing their code and models.

## License

The ReT-UAP code is released under the [MIT License](LICENSE). The vendored third-party
code keeps its original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
