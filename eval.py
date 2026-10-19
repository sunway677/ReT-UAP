"""
Evaluation of ReT-UAP universal perturbations on image-text retrieval
(manuscript Sec. 5.1, "Evaluation Metric"; protocol of C-PGC).

Usage (from the repository root):
    python eval.py --config ./configs/Retrieval_flickr_test.yaml \
        --source_model ALBEF --checkpoint_dir ./checkpoint \
        --load_dir output_retuap --epoch 99 --output retuap_flickr30k

Test splits: Flickr30K 1K (1,000 images / 5,000 captions,
configs/Retrieval_flickr_test.yaml) and MS-COCO 5K (5,000 images / 25,010
captions, configs/Retrieval_coco_test.yaml).

Inputs
    <load_dir>/<TAG>/<DATASET>/uap_noise-<epoch>.pth      delta_x*, (1, 3, H, W)
    <load_dir>/<TAG>/<DATASET>/uap_embedding-<epoch>.pth  delta_t*, (K_p, d_tok)
    <word_dir>/word_embedding_<TAG>_<DATASET>.pth         token-embedding table of the source model
    <word_dir>/word_idx_BERT.npy or word_idx_CLIP.npy     whole-word vocabulary filter of its family
TAG is the source model name with '/' replaced by '_' (ViT-B/16 -> ViT-B_16)
and DATASET is config['dataset'] (flickr30k or mscoco). delta_x* and delta_t*
are the perturbations written by retuap_train.py at the end of training.

Models
    Source: --source_model, the primary surrogate the perturbations were
    trained on. Targets: all six models ALBEF, TCL, BLIP, XVLM, ViT-B/16 and
    ViT-B/32 (fine-tuned retrieval checkpoints <checkpoint_dir>/<MODEL>/<DATASET>.pth
    for ALBEF/TCL/BLIP/XVLM, OpenAI weights for CLIP); the source column is the
    white-box result. Models are loaded one at a time.

Adversarial inputs (x + delta_x*, T (+) delta_t*)
    Image. Images are first brought to the source model's input space, as in
    the C-PGC evaluation:
        BERT-family source (ALBEF, TCL, BLIP, XVLM): Resize((image_res, image_res)), bicubic;
        CLIP source (ViT-B/16, ViT-B/32):            Resize(224) + CenterCrop(224), bicubic.
    The full 3-channel delta_x* is broadcast over the batch and added, and
    the result is clamped to [0, 1]. If the spatial size of delta_x* differs
    from this input space (e.g. a 384 x 384 delta_x* of a CLIP primary on the
    224 x 224 CLIP input), delta_x* is first resized bilinearly, as the
    training code does for CLIP surrogates. Targets of the source's family use
    this image directly; targets of the other family receive it through
    ToPILImage and a resize: CLIP targets of a BERT-family source get
    Resize(224) + CenterCrop(224), BERT-family targets of a CLIP source get
    Resize((image_res, image_res)). Every model normalizes with the CLIP
    mean / std.
    Text. Each row of delta_t* is mapped to the nearest word of the source
    model's own token-embedding table restricted to its family's whole-word
    filter, and the word(s) replace the most important word(s) of every
    caption, word importance being measured with the source model
    (attacker.py). The adversarial captions are built ONCE, before any target
    is evaluated, and the same captions are fed to every target model.

Retrieval protocol (C-PGC: both sides perturbed)
    Every image and every caption of the test split is replaced by its
    adversarial version. In TR (image-to-text, column '<target> i2t') the query
    image and the caption gallery are adversarial; in IR (text-to-image,
    column '<target> t2i') the query caption and the image gallery are
    adversarial. The clean reference is the same computation with
    delta_x* = 0 and the original captions (same preprocessing path, same
    query order).
    Scores: ALBEF, TCL, BLIP and XVLM rank by ITC similarity and re-rank the
    top k_test candidates of every query with the ITM head (k_test from the
    config: 128 for Flickr30K, 256 for MS-COCO; other candidates get -100);
    BLIP's ITM input starts with its [ENC] token, as in BLIP's own retrieval
    evaluation. ViT-B/16 and ViT-B/32 rank by cosine similarity (captions
    tokenized with clip.tokenize, 77 tokens, truncated).
    Ranks: a TR query (image) has the best rank among its ground-truth
    captions; an IR query (caption) has the rank of its image.

Attack success rate (for every target model, direction and K in {1, 5, 10})
    C_K = clean queries with rank < K          (clean-correct set)
    F_K = adversarial queries with rank >= K   (adversarially failed set)
    ASR@K = 100 * |C_K & F_K| / |C_K|
    C_K is computed on the fly for each target model from its own clean
    scores, in the same query order as the adversarial pass. The manuscript
    reports ASR@1 for TR and IR.

Output
    A table printed to stdout and outputs/<output>/<TAG>_<DATASET>.xlsx with
      sheet 'ASR':    rows R@1, R@5, R@10; columns '<target> i2t' (TR ASR) and
                      '<target> t2i' (IR ASR), in percent (C-PGC layout);
      sheet 'Recall': clean and adversarial Recall@K and |C_K| per target;
      sheet 'Info':   source, perturbation files and adversarial words.

Fixes with respect to the research evaluation script:
  - the clean-correct sets C_K are recomputed for every target instead of
    being read from precomputed std_eval_idx files (which were misaligned with
    the query order for Flickr30K, and ViT-B/32 used the ViT-B/16 set); being
    computed through the same resize path as the adversarial inputs, they also
    no longer count resize-induced misses as attack successes;
  - the nearest word is looked up in the source model's own token-embedding
    table (the research files held the ALBEF table for every BERT-family
    source and the ViT-B/16 table for both CLIP sources);
  - CLIP candidates are restricted to whole-word BPE tokens and byte-decoded,
    and CLIP-source word importance is computed in CLIP BPE token space;
  - BLIP's ITM re-ranking starts the caption with [ENC] instead of [CLS].
"""

import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from attacker import (BERT_FAMILY, CLIP_FAMILY, MAX_TEXT_LEN, TextAttacker,
                      decode_bert_token, decode_clip_token, embedding_to_words,
                      model_family)

TARGET_MODELS = list(BERT_FAMILY + CLIP_FAMILY)
CLIP_RES = 224
RECALL_KS = (1, 5, 10)
BICUBIC = transforms.InterpolationMode.BICUBIC

# Input normalization used for every model (CLIP mean / std, as in C-PGC).
IMAGE_NORMALIZE = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))


def model_tag(name):
    """File-system tag of a model name ('ViT-B/16' -> 'ViT-B_16')."""
    return name.replace('/', '_')


# ==========================================
# Arguments
# ==========================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='ASR of ReT-UAP perturbations (x + delta_x*, T (+) delta_t*) on '
                    'image-text retrieval against ALBEF, TCL, BLIP, XVLM, ViT-B/16 and ViT-B/32.')
    parser.add_argument('--config', default='./configs/Retrieval_flickr_test.yaml',
                        help='test-split config (annotation file, image root, image_res, k_test)')
    parser.add_argument('--source_model', required=True, choices=TARGET_MODELS,
                        help='primary surrogate the perturbations were trained on')
    parser.add_argument('--checkpoint_dir', '--checkpoint', dest='checkpoint_dir', default='./checkpoint',
                        help='directory with <MODEL>/<dataset>.pth retrieval checkpoints')
    parser.add_argument('--load_dir', required=True,
                        help='training save_dir; perturbations are read from <load_dir>/<TAG>/<dataset>/')
    parser.add_argument('--epoch', type=int, default=99,
                        help='epoch index of uap_noise-<epoch>.pth / uap_embedding-<epoch>.pth')
    parser.add_argument('--word_dir', default='.',
                        help='directory with word_embedding_<TAG>_<dataset>.pth and word_idx_{BERT,CLIP}.npy')
    parser.add_argument('--text_encoder', default='bert-base-uncased',
                        help='HuggingFace name of the BERT tokenizer of the BERT-family models')
    parser.add_argument('--batch_size', type=int, default=16, help='images per evaluation batch')
    parser.add_argument('--num_workers', type=int, default=4, help='data-loader workers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='retuap',
                        help='results are written to outputs/<output>/')
    return parser.parse_args(argv)


# ==========================================
# Perturbations and vocabulary
# ==========================================

def _load_tensor(path):
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # torch without the weights_only argument
        obj = torch.load(path, map_location='cpu')
    if not torch.is_tensor(obj):
        raise TypeError(f'{path} does not contain a tensor')
    return obj


def load_perturbations(load_dir, source_model, dataset, epoch):
    """delta_x* as (1, 3, H, W) and delta_t* as (K_p, d), both float32."""
    run_dir = os.path.join(load_dir, model_tag(source_model), dataset)
    noise_path = os.path.join(run_dir, f'uap_noise-{epoch}.pth')
    embedding_path = os.path.join(run_dir, f'uap_embedding-{epoch}.pth')
    for path in (noise_path, embedding_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found (written by retuap_train.py)')
    delta_x = _load_tensor(noise_path).float()
    if delta_x.dim() == 3:
        delta_x = delta_x.unsqueeze(0)
    if delta_x.dim() != 4 or delta_x.shape[0] != 1 or delta_x.shape[1] != 3:
        raise ValueError(f'{noise_path}: expected a (1, 3, H, W) image UAP, got {tuple(delta_x.shape)}')
    delta_t = _load_tensor(embedding_path).float()
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(0)
    if delta_t.dim() != 2:
        raise ValueError(f'{embedding_path}: expected a (K_p, d) text UAP, got {tuple(delta_t.shape)}')
    return delta_x, delta_t, run_dir


def load_word_vocabulary(word_dir, source_model, dataset):
    """Token-embedding table of the source model and its family's whole-word filter."""
    table_path = os.path.join(word_dir, f'word_embedding_{model_tag(source_model)}_{dataset}.pth')
    idx_name = 'word_idx_BERT.npy' if model_family(source_model) == 'bert' else 'word_idx_CLIP.npy'
    idx_path = os.path.join(word_dir, idx_name)
    if not os.path.exists(table_path):
        raise FileNotFoundError(f'{table_path} not found; run scripts/generate_word_embeddings.py')
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f'{idx_path} not found (shipped in the repository root)')
    return _load_tensor(table_path).float(), np.load(idx_path), table_path, idx_path


def token_decoder(source_model, tokenizer):
    """token id -> whole word (or None) in the source model's vocabulary."""
    if model_family(source_model) == 'bert':
        return lambda token_id: decode_bert_token(token_id, tokenizer)
    return decode_clip_token


@torch.no_grad()
def build_adversarial_captions(attacker, source_net, captions, chunk_size=256):
    """T (+) delta_t* for every caption, built once with the source model."""
    adv = []
    for s in tqdm(range(0, len(captions), chunk_size), desc='adversarial captions'):
        adv += attacker.get_adv_text(source_net, captions[s:s + chunk_size])
    return adv


# ==========================================
# Images
# ==========================================

def build_transforms(source_model, image_res, clip_res=CLIP_RES):
    """(source_transform, cross_transform). source_transform maps a PIL image to
    the source model's input space; cross_transform maps a [0, 1] tensor of
    that space to the input space of a target of the other family."""
    if model_family(source_model) == 'bert':
        source = transforms.Compose([
            transforms.Resize((image_res, image_res), interpolation=BICUBIC),
            transforms.ToTensor()])
        cross = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(clip_res, interpolation=BICUBIC),
            transforms.CenterCrop(clip_res),
            transforms.ToTensor()])
    else:
        source = transforms.Compose([
            transforms.Resize(clip_res, interpolation=BICUBIC),
            transforms.CenterCrop(clip_res),
            transforms.ToTensor()])
        cross = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_res, image_res), interpolation=BICUBIC),
            transforms.ToTensor()])
    return source, cross


def apply_uap(images, delta_x):
    """clamp(x + delta_x, 0, 1) for a batch x (B, 3, H, W) in [0, 1]. The full
    3-channel delta_x (1, 3, h, w) is broadcast over the batch; it is resized
    bilinearly to (H, W) if (h, w) differs. delta_x = None returns x."""
    if delta_x is None:
        return images
    d = delta_x.to(device=images.device, dtype=images.dtype)
    if d.dim() == 3:
        d = d.unsqueeze(0)
    if d.shape[0] != 1 or d.shape[1] != images.shape[1]:
        raise ValueError(f'image UAP of shape {tuple(d.shape)} does not match images {tuple(images.shape)}')
    if d.shape[-2:] != images.shape[-2:]:
        d = F.interpolate(d, size=tuple(images.shape[-2:]), mode='bilinear', align_corners=False)
    return torch.clamp(images + d, 0.0, 1.0)


# ==========================================
# Encoding and scoring
# ==========================================

@torch.no_grad()
def encode_test_set(target_name, model, tokenizer, data_loader, captions, delta_x,
                    source_family, cross_transform, device):
    """One pass of one target model over the test split.

    captions : list of caption strings indexed by caption id (the original or
               the adversarial captions)
    delta_x  : image UAP added in the source input space, or None (clean)
    Returns the features (and ITM inputs for BERT-family targets) indexed by
    image id and caption id.
    """
    family = model_family(target_name)
    num_image, num_text = len(data_loader.dataset), len(data_loader.dataset.text)
    rec = {}

    def store(key, ids, value, n):
        value = value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
        if key not in rec:
            rec[key] = torch.zeros((n,) + tuple(value.shape[1:]), dtype=value.dtype)
        rec[key][ids] = value

    for images, _, image_ids, text_id_groups in data_loader:
        text_ids = [t for group in text_id_groups for t in group]
        texts = [captions[t] for t in text_ids]
        images = apply_uap(images.to(device), delta_x)
        if family != source_family:
            images = torch.stack([cross_transform(img.cpu()) for img in images], dim=0).to(device)
        images = IMAGE_NORMALIZE(images)
        if family == 'clip':
            out = model.inference(images, texts)
            store('image_feats', image_ids, out['image_feat'], num_image)
            store('text_feats', text_ids, out['text_feat'], num_text)
        else:
            text_input = tokenizer(texts, padding='max_length', truncation=True,
                                   max_length=MAX_TEXT_LEN, return_tensors='pt').to(device)
            out_img = model.inference_image(images)
            out_txt = model.inference_text(text_input)
            store('image_feats', image_ids, out_img['image_feat'], num_image)
            store('image_embeds', image_ids, out_img['image_embed'], num_image)
            store('text_feats', text_ids, out_txt['text_feat'], num_text)
            store('text_atts', text_ids, text_input.attention_mask, num_text)
            if target_name == 'BLIP':
                # BLIP's ITM re-encodes the token ids (with [ENC] first).
                store('text_ids', text_ids, text_input.input_ids, num_text)
            else:
                # ALBEF / TCL / XVLM fuse the unimodal text embeddings.
                store('text_embeds', text_ids, out_txt['text_embed'], num_text)
    return rec


def _blip_enc_token_id(model, tokenizer):
    for owner in (tokenizer, getattr(model, 'tokenizer', None)):
        token_id = getattr(owner, 'enc_token_id', None)
        if token_id is not None:
            return int(token_id)
    raise AttributeError('BLIP tokenizer has no enc_token_id')


@torch.no_grad()
def score_matrices(target_name, model, tokenizer, rec, k_test, device):
    """(scores_i2t (N_img, N_txt), scores_t2i (N_txt, N_img)) as numpy arrays."""
    image_feats, text_feats = rec['image_feats'], rec['text_feats']
    sims = image_feats @ text_feats.t()
    if model_family(target_name) == 'clip':
        return sims.numpy(), sims.t().contiguous().numpy()

    num_image, num_text = sims.shape
    image_embeds, text_atts = rec['image_embeds'], rec['text_atts']
    text_embeds, text_ids = rec.get('text_embeds'), rec.get('text_ids')
    if target_name == 'BLIP':
        text_ids = text_ids.clone()
        text_ids[:, 0] = _blip_enc_token_id(model, tokenizer)

    def itm(txt_idx, img_embeds, n):
        img_att = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=device)
        if target_name == 'BLIP':
            out = model.text_encoder(text_ids[txt_idx].to(device),
                                     attention_mask=text_atts[txt_idx].to(device),
                                     encoder_hidden_states=img_embeds,
                                     encoder_attention_mask=img_att, return_dict=True)
        else:
            out = model.text_encoder(encoder_embeds=text_embeds[txt_idx].to(device),
                                     attention_mask=text_atts[txt_idx].to(device),
                                     encoder_hidden_states=img_embeds,
                                     encoder_attention_mask=img_att, return_dict=True, mode='fusion')
        return model.itm_head(out.last_hidden_state[:, 0, :])[:, 1].float().cpu()

    k_i2t = min(k_test, num_text)
    scores_i2t = torch.full((num_image, num_text), -100.0)
    for i in tqdm(range(num_image), desc=f'{target_name} ITM i2t', leave=False):
        _, topk_idx = sims[i].topk(k=k_i2t, dim=0)
        img = image_embeds[i].unsqueeze(0).repeat(k_i2t, 1, 1).to(device)
        scores_i2t[i, topk_idx] = itm(topk_idx, img, k_i2t)

    k_t2i = min(k_test, num_image)
    sims_t = sims.t()
    scores_t2i = torch.full((num_text, num_image), -100.0)
    for t in tqdm(range(num_text), desc=f'{target_name} ITM t2i', leave=False):
        _, topk_idx = sims_t[t].topk(k=k_t2i, dim=0)
        img = image_embeds[topk_idx].to(device)
        txt_idx = torch.full((k_t2i,), t, dtype=torch.long)
        scores_t2i[t, topk_idx] = itm(txt_idx, img, k_t2i)
    return scores_i2t.numpy(), scores_t2i.numpy()


# ==========================================
# Ranks and ASR
# ==========================================

def retrieval_ranks(scores_i2t, scores_t2i, img2txt, txt2img):
    """0-based rank of every query (C-PGC ranking: descending np.argsort).
    TR (image i): best rank among its captions img2txt[i]; IR (caption t):
    rank of its image txt2img[t]."""
    ranks_i2t = np.zeros(scores_i2t.shape[0], dtype=np.int64)
    for i, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        pos = np.empty(len(inds), dtype=np.int64)
        pos[inds] = np.arange(len(inds))
        ranks_i2t[i] = pos[list(img2txt[i])].min()
    ranks_t2i = np.zeros(scores_t2i.shape[0], dtype=np.int64)
    for t, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks_t2i[t] = int(np.where(inds == txt2img[t])[0][0])
    return ranks_i2t, ranks_t2i


def attack_success_rate(clean_ranks, adv_ranks, k):
    """(ASR@k in percent, |C_k|, |C_k & F_k|): C_k = clean queries with
    rank < k, F_k = adversarial queries with rank >= k, same query order."""
    clean_ranks, adv_ranks = np.asarray(clean_ranks), np.asarray(adv_ranks)
    if clean_ranks.shape != adv_ranks.shape:
        raise ValueError('clean and adversarial ranks must cover the same queries')
    clean_correct = clean_ranks < k
    flipped = clean_correct & (adv_ranks >= k)
    n_correct, n_flipped = int(clean_correct.sum()), int(flipped.sum())
    asr = round(100.0 * n_flipped / n_correct, 2) if n_correct > 0 else float('nan')
    return asr, n_correct, n_flipped


def evaluate_target(target_name, model, tokenizer, data_loader, adv_captions, delta_x,
                    source_family, cross_transform, k_test, device):
    """Clean and adversarial pass of one target; ranks and ASR@K."""
    dataset = data_loader.dataset
    ranks = {}
    for condition, captions, uap in (('clean', list(dataset.text), None),
                                     ('adv', adv_captions, delta_x)):
        rec = encode_test_set(target_name, model, tokenizer, data_loader, captions, uap,
                              source_family, cross_transform, device)
        s_i2t, s_t2i = score_matrices(target_name, model, tokenizer, rec, k_test, device)
        del rec
        ranks[condition] = retrieval_ranks(s_i2t, s_t2i, dataset.img2txt, dataset.txt2img)
    result = {'ranks': ranks}
    for d, direction in enumerate(('TR', 'IR')):
        clean_r, adv_r = ranks['clean'][d], ranks['adv'][d]
        for k in RECALL_KS:
            asr, n_correct, n_flipped = attack_success_rate(clean_r, adv_r, k)
            result[f'{direction}_ASR@{k}'] = asr
            result[f'{direction}_clean_correct@{k}'] = n_correct
            result[f'{direction}_flipped@{k}'] = n_flipped
            result[f'{direction}_clean_R@{k}'] = round(100.0 * float(np.mean(clean_r < k)), 2)
            result[f'{direction}_adv_R@{k}'] = round(100.0 * float(np.mean(adv_r < k)), 2)
    return result


def run_evaluation(source_model, get_model, release_model, data_loader, adv_captions, delta_x,
                   cross_transform, k_test, device, target_models=None):
    """Evaluate every target model on the same adversarial captions and image UAP.

    get_model(name) -> (model, tokenizer); release_model(name) frees it.
    """
    target_models = list(target_models or TARGET_MODELS)
    if len(adv_captions) != len(data_loader.dataset.text):
        raise ValueError('one adversarial caption per test caption is required')
    source_family = model_family(source_model)
    results = {}
    for name in target_models:
        model, tokenizer = get_model(name)
        start = time.time()
        results[name] = evaluate_target(name, model, tokenizer, data_loader, adv_captions, delta_x,
                                        source_family, cross_transform, k_test, device)
        release_model(name)
        r = results[name]
        print(f'[{name}] TR ASR@1 {r["TR_ASR@1"]:.2f}  IR ASR@1 {r["IR_ASR@1"]:.2f}  '
              f'(clean R@1 {r["TR_clean_R@1"]:.2f}/{r["IR_clean_R@1"]:.2f}, '
              f'adv R@1 {r["TR_adv_R@1"]:.2f}/{r["IR_adv_R@1"]:.2f}, '
              f'{datetime.timedelta(seconds=int(time.time() - start))})', flush=True)
    return results


# ==========================================
# Reporting
# ==========================================

def format_table(source_model, dataset, results):
    lines = [f'ASR (%) of ReT-UAP, source {source_model}, {dataset} '
             f'(TR = image-to-text, IR = text-to-image; R@1 = clean / adversarial Recall@1)',
             f'{"Target":<10} {"TR ASR@1":>9} {"IR ASR@1":>9}   {"TR R@1 clean/adv":>17} {"IR R@1 clean/adv":>17}']
    for name, r in results.items():
        tag = name + (' *' if name == source_model else '')
        lines.append(f'{tag:<10} {r["TR_ASR@1"]:>9.2f} {r["IR_ASR@1"]:>9.2f}   '
                     f'{r["TR_clean_R@1"]:>8.2f}/{r["TR_adv_R@1"]:<8.2f} '
                     f'{r["IR_clean_R@1"]:>8.2f}/{r["IR_adv_R@1"]:<8.2f}')
    lines.append('* white-box (source) model')
    return '\n'.join(lines)


def save_results(path, results, info):
    import pandas as pd

    asr = {}
    for name, r in results.items():
        asr[f'{name} i2t'] = [r[f'TR_ASR@{k}'] for k in RECALL_KS]
        asr[f'{name} t2i'] = [r[f'IR_ASR@{k}'] for k in RECALL_KS]
    asr_df = pd.DataFrame(asr, index=[f'R@{k}' for k in RECALL_KS])

    rows = []
    for name, r in results.items():
        for direction in ('TR', 'IR'):
            row = {'target': name, 'direction': direction}
            for k in RECALL_KS:
                row[f'clean R@{k}'] = r[f'{direction}_clean_R@{k}']
                row[f'adv R@{k}'] = r[f'{direction}_adv_R@{k}']
                row[f'|C@{k}|'] = r[f'{direction}_clean_correct@{k}']
                row[f'flipped@{k}'] = r[f'{direction}_flipped@{k}']
            rows.append(row)
    recall_df = pd.DataFrame(rows)
    info_df = pd.DataFrame({'key': list(info.keys()), 'value': [str(v) for v in info.values()]})

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        asr_df.to_excel(writer, sheet_name='ASR')
        recall_df.to_excel(writer, sheet_name='Recall', index=False)
        info_df.to_excel(writer, sheet_name='Info', index=False)
    return path


# ==========================================
# Entry point
# ==========================================

def load_eval_model(name, args, config, device):
    """(model, tokenizer) of one model, in float32 eval mode."""
    from utils import load_model

    if name in CLIP_FAMILY:
        ckpt = ''
    else:
        ckpt = os.path.join(args.checkpoint_dir, name, f'{config["dataset"]}.pth')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'{ckpt} not found')
    model, ref_model, tokenizer = load_model(name, ckpt, args.text_encoder, config, device)
    del ref_model
    model = model.float().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def main(argv=None):
    args = parse_args(argv)
    from ruamel.yaml import YAML
    from dataset import paired_dataset

    with open(args.config, 'r') as f:
        config = YAML(typ='safe').load(f)
    dataset_name = config['dataset']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('[warn] CUDA is not available; evaluating on CPU will be very slow')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    start_time = time.time()

    # Perturbations and the source model's vocabulary.
    delta_x, delta_t, run_dir = load_perturbations(args.load_dir, args.source_model, dataset_name, args.epoch)
    word_table, word_idx, table_path, idx_path = load_word_vocabulary(args.word_dir, args.source_model,
                                                                      dataset_name)
    print(f'delta_x {tuple(delta_x.shape)} (L_inf {delta_x.abs().max().item() * 255:.2f}/255), '
          f'delta_t {tuple(delta_t.shape)}, from {run_dir}')

    source_transform, cross_transform = build_transforms(args.source_model, config['image_res'])
    test_dataset = paired_dataset(config['annotation_file'], source_transform, config['image_root'])
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)
    print(f'{dataset_name} test split: {len(test_dataset)} images, {len(test_dataset.text)} captions')

    # Adversarial words and captions, built once with the source model.
    cache = {}
    source_net, source_tokenizer = load_eval_model(args.source_model, args, config, device)
    cache[args.source_model] = (source_net, source_tokenizer)
    adv_words, adv_ids = embedding_to_words(delta_t, word_table, word_idx,
                                            token_decoder(args.source_model, source_tokenizer))
    print(f'adversarial words: {adv_words} (token ids {adv_ids})')
    attacker = TextAttacker(args.source_model, source_tokenizer, adv_words)
    adv_captions = build_adversarial_captions(attacker, source_net, list(test_dataset.text))
    n_changed = sum(a != c for a, c in zip(adv_captions, test_dataset.text))
    print(f'{n_changed}/{len(adv_captions)} captions modified; e.g. '
          f'"{test_dataset.text[0]}" -> "{adv_captions[0]}"')
    del source_net

    def get_model(name):
        if name in cache:
            return cache.pop(name)
        return load_eval_model(name, args, config, device)

    def release_model(name):
        cache.pop(name, None)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    results = run_evaluation(args.source_model, get_model, release_model, data_loader, adv_captions,
                             delta_x, cross_transform, config['k_test'], device)

    print(format_table(args.source_model, dataset_name, results))
    info = {'source_model': args.source_model, 'dataset': dataset_name, 'epoch': args.epoch,
            'uap_noise': os.path.join(run_dir, f'uap_noise-{args.epoch}.pth'),
            'uap_embedding': os.path.join(run_dir, f'uap_embedding-{args.epoch}.pth'),
            'word_embedding': table_path, 'word_idx': idx_path,
            'adversarial_words': ' '.join(adv_words), 'k_test': config['k_test'],
            'captions_modified': f'{n_changed}/{len(adv_captions)}'}
    save_path = os.path.join('outputs', args.output, f'{model_tag(args.source_model)}_{dataset_name}.xlsx')
    save_results(save_path, results, info)
    print(f'Saved to {save_path}')
    print(f'Evaluate time {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == '__main__':
    main()
