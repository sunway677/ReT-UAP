"""
Write the token-embedding tables read by retuap_train.py and eval.py.

For every surrogate MODEL in {ALBEF, TCL, BLIP, XVLM, ViT-B/16, ViT-B/32} and
every requested DATASET in {flickr30k, mscoco} this script writes

    <out_dir>/word_embedding_<TAG>_<DATASET>.pth      (TAG = MODEL with '/' -> '_')

which holds the (V, d) float32 token-embedding table of that model:

- ALBEF, TCL, BLIP, XVLM: 'text_encoder.embeddings.word_embeddings.weight' of
  the model's own fine-tuned checkpoint <checkpoint_dir>/<MODEL>/<DATASET>.pth
  (30522 x 768; 30524 x 768 for BLIP, whose tokenizer adds [DEC] and [ENC]).
- ViT-B/16, ViT-B/32: token_embedding.weight of the OpenAI CLIP model
  (49408 x 512), loaded with models.clip_model.clip.load on the CPU. CLIP is
  not fine-tuned, so both datasets get the same table.

retuap_train.py uses the primary surrogate's table for the L2-norm range of the
text perturbation delta_t; eval.py maps every row of delta_t to the nearest
token of the source model's table, restricted to the whole-word vocabulary
filter of its family:

- word_idx_BERT.npy (shipped): 17,952 bert-base-uncased ids of whole words made
  of lowercase ASCII letters (the vocabulary filter of the C-PGC evaluation
  code); used for ALBEF, TCL, BLIP and XVLM. It is not regenerated here.
- word_idx_CLIP.npy (shipped): ids of the CLIP BPE tokens '<w></w>' whose word
  w consists of lowercase ASCII letters only and re-encodes to exactly that
  single token (32,875 ids), i.e. the CLIP counterpart of word_idx_BERT.npy;
  used for ViT-B/16 and ViT-B/32. It is derived deterministically from the
  BPE vocabulary in models/clip_model/ and can be rebuilt with
  --write_word_idx_clip.

Usage (from the repository root, after downloading the checkpoints):
    python scripts/generate_word_embeddings.py --all
    python scripts/generate_word_embeddings.py --dataset flickr30k --models ALBEF ViT-B/16
    python scripts/generate_word_embeddings.py --write_word_idx_clip
The CLIP weights are downloaded to ~/.cache/clip on first use (or read from
--clip_download_root). Everything runs on the CPU.
"""
import argparse
import os
import re
import sys
import zipfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

BERT_MODELS = ('ALBEF', 'TCL', 'BLIP', 'XVLM')
CLIP_MODELS = ('ViT-B/16', 'ViT-B/32')
ALL_MODELS = BERT_MODELS + CLIP_MODELS
DATASETS = ('flickr30k', 'mscoco')

# Key of the text-encoder token-embedding table in the fine-tuned checkpoints.
# utils.load_model strips a 'bert.' infix from text-encoder keys, so a
# checkpoint that stores the table under the second name is accepted too.
BERT_TABLE_KEYS = ('text_encoder.embeddings.word_embeddings.weight',
                   'text_encoder.bert.embeddings.word_embeddings.weight')

CLIP_WORD_END = '</w>'
CLIP_WORD_RE = re.compile(r'[a-z]+')


def model_tag(name):
    return name.replace('/', '_')


def _torch_load_cpu(path):
    """torch.load on the CPU; zip-format checkpoints are memory-mapped so that
    large files (e.g. the 3.5 GB TCL checkpoint with optimizer state) are not
    read into RAM."""
    kwargs = {'map_location': 'cpu'}
    if zipfile.is_zipfile(path):
        kwargs['mmap'] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except Exception:
        # The checkpoints also store configs / optimizer states, which the
        # restricted unpickler may reject. Only load checkpoints you trust.
        return torch.load(path, weights_only=False, **kwargs)


def bert_word_table(ckpt_path):
    """(V, 768) float32 token-embedding table of a fine-tuned ALBEF/TCL/BLIP/XVLM checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'{ckpt_path} not found (expected <checkpoint_dir>/<MODEL>/<DATASET>.pth)')
    checkpoint = _torch_load_cpu(ckpt_path)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    for key in BERT_TABLE_KEYS:
        if key in state_dict:
            table = state_dict[key].detach().to(torch.float32).clone().contiguous()
            del checkpoint, state_dict
            return table, key
    similar = [k for k in state_dict if 'word_embeddings' in k]
    raise KeyError(f'{ckpt_path}: none of {BERT_TABLE_KEYS} found; word-embedding keys present: {similar}')


def clip_word_table(name, download_root=None):
    """(49408, 512) float32 token-embedding table of an OpenAI CLIP model."""
    from models.clip_model import clip
    model, _ = clip.load(name, device='cpu', jit=False, download_root=download_root)
    table = model.token_embedding.weight.detach().to(torch.float32).clone().contiguous()
    del model
    return table


def clip_whole_word_ids():
    """Sorted ids of the CLIP BPE tokens '<w></w>' with w in [a-z]+ that
    re-encode to exactly that single token (deterministic)."""
    from models.clip_model.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    ids = []
    for token_id in sorted(tokenizer.decoder):
        token = tokenizer.decoder[token_id]
        if not token.endswith(CLIP_WORD_END):
            continue
        word = token[:-len(CLIP_WORD_END)]
        # ASCII letters map to themselves in CLIP's byte-to-unicode alphabet,
        # so the BPE string of such a token is the word itself.
        if not CLIP_WORD_RE.fullmatch(word):
            continue
        if tokenizer.encode(word) != [token_id]:
            continue
        ids.append(token_id)
    return np.asarray(ids, dtype=np.int64)


def describe(table):
    norms = table.norm(dim=1)
    return (f'shape={tuple(table.shape)} dtype={str(table.dtype).replace("torch.", "")} '
            f'row-norm min={norms.min().item():.4f} max={norms.max().item():.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', choices=DATASETS, default=None,
                        help='dataset(s) whose fine-tuned checkpoints are used')
    parser.add_argument('--all', action='store_true', help='both datasets (flickr30k and mscoco)')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, default=list(ALL_MODELS),
                        help='surrogates to process (default: all six)')
    parser.add_argument('--checkpoint_dir', default=os.path.join(REPO_ROOT, 'checkpoint'),
                        help='directory with <MODEL>/<DATASET>.pth for ALBEF, TCL, BLIP and XVLM')
    parser.add_argument('--out_dir', default=REPO_ROOT,
                        help='output directory (retuap_train.py and eval.py read the repository root)')
    parser.add_argument('--clip_download_root', default=None,
                        help='CLIP weight cache (default: ~/.cache/clip)')
    parser.add_argument('--write_word_idx_clip', action='store_true',
                        help='also rebuild <out_dir>/word_idx_CLIP.npy from the CLIP BPE vocabulary')
    args = parser.parse_args()
    if args.all:
        args.dataset = list(DATASETS)
    if not args.dataset and not args.write_word_idx_clip:
        parser.error('pass --all, --dataset <name> ..., and/or --write_word_idx_clip')
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.write_word_idx_clip:
        ids = clip_whole_word_ids()
        path = os.path.join(args.out_dir, 'word_idx_CLIP.npy')
        np.save(path, ids)
        print(f'[word_idx] {len(ids)} CLIP whole-word ids (range {ids.min()}..{ids.max()}) -> {path}')

    clip_tables = {}
    for dataset in args.dataset or []:
        for name in args.models:
            if name in BERT_MODELS:
                ckpt = os.path.join(args.checkpoint_dir, name, f'{dataset}.pth')
                table, key = bert_word_table(ckpt)
                source = f'{ckpt} [{key}]'
            else:
                if name not in clip_tables:  # identical for both datasets
                    clip_tables[name] = clip_word_table(name, args.clip_download_root)
                table = clip_tables[name]
                source = f'CLIP {name} token_embedding.weight'
            path = os.path.join(args.out_dir, f'word_embedding_{model_tag(name)}_{dataset}.pth')
            torch.save(table, path)
            print(f'[{name} / {dataset}] {describe(table)} <- {source} -> {path}')


if __name__ == '__main__':
    main()
