"""
Text side of the ReT-UAP evaluation (used by eval.py).

The text UAP delta_t* (K_p rows in the token-embedding space of the source,
i.e. primary, surrogate) is realized at test time as word substitutions, as
in C-PGC (the manuscript: "delta_t is mapped back to the vocabulary space and
realized through token-level word substitution"):

1. embedding_to_words: every row of delta_t* is mapped to the nearest (L2)
   entry of the source model's own token-embedding table, restricted to the
   whole-word vocabulary filter of its family (word_idx_BERT.npy for ALBEF,
   TCL, BLIP and XVLM; word_idx_CLIP.npy for ViT-B/16 and ViT-B/32). Candidates
   that do not decode to a valid whole word, or that are stop words, are
   skipped and the next nearest entry is taken.
     BERT family: the token string itself; it must be alphabetic and must
       re-tokenize to exactly itself (no '##' pieces, no special tokens).
     CLIP: the BPE token must end in '</w>' (a whole word); the marker is
       stripped and the byte-level BPE string is decoded to UTF-8; the word
       must be alphabetic and must re-encode to exactly the same single CLIP
       token, so the inserted word is the token delta_t* was mapped to.

2. TextAttacker.get_adv_text: in every caption the most important
   space-separated words are replaced by these words, at most one word per
   row of delta_t* (K_p words per caption). Word importance is the KL
   divergence between the softmax of the source model's clean (normalized)
   text feature and that of the feature with the word masked (C-PGC):
     BERT family: the word is replaced by '[UNK]' and the caption is
       re-tokenized with the source model's BERT tokenizer (30 tokens); a word
       is selectable if its first word piece lies inside the 30-token caption.
     CLIP: importance is computed in CLIP BPE token space: the word's BPE
       tokens are replaced by the single special token <|startoftext|> (CLIP
       has no [UNK]/[MASK] token; this is the masking token used in training)
       and the caption is encoded with the source CLIP text encoder (77-token
       context); a word is selectable if its first BPE token lies before
       <|endoftext|> in the 77-token context.
   Stop words (utils.get_filter_words) are never replaced. Ties in importance
   keep the earlier word.

The adversarial captions depend only on the caption and the source model, so
eval.py builds them once and feeds the same captions to every target model.
"""

import copy

import torch
import torch.nn.functional as F


BERT_FAMILY = ('ALBEF', 'TCL', 'BLIP', 'XVLM')
CLIP_FAMILY = ('ViT-B/16', 'ViT-B/32')

MAX_TEXT_LEN = 30                     # BERT-family caption length in tokens, as in C-PGC
CLIP_CONTEXT_LENGTH = 77
CLIP_MASK_TOKEN = '<|startoftext|>'   # same masking token as the training code
CLIP_WORD_END = '</w>'

_FILTER_WORDS = None
_CLIP_TOKENIZER = None


def model_family(name):
    if name in BERT_FAMILY:
        return 'bert'
    if name in CLIP_FAMILY:
        return 'clip'
    raise ValueError(f'unknown model {name!r}; expected one of {BERT_FAMILY + CLIP_FAMILY}')


def get_filter_words():
    """Stop words and punctuation that are never inserted nor replaced (C-PGC list)."""
    global _FILTER_WORDS
    if _FILTER_WORDS is None:
        from utils import get_filter_words as _get
        _FILTER_WORDS = _get()
    return _FILTER_WORDS


def get_clip_tokenizer():
    """The CLIP byte-level BPE tokenizer (models/clip_model/simple_tokenizer.py)."""
    global _CLIP_TOKENIZER
    if _CLIP_TOKENIZER is None:
        from models.clip_model.simple_tokenizer import SimpleTokenizer
        _CLIP_TOKENIZER = SimpleTokenizer()
    return _CLIP_TOKENIZER


# ==========================================
# delta_t -> vocabulary words
# ==========================================

def decode_bert_token(token_id, tokenizer):
    """Whole word of a BERT vocabulary id, or None if the id is not a whole word."""
    word = tokenizer.convert_ids_to_tokens(int(token_id))
    if not isinstance(word, str) or not word.isalpha():
        return None  # excludes '##' pieces, [CLS]/[SEP]/[PAD]/[UNK], digits, punctuation
    if tokenizer.tokenize(word) != [word]:
        return None
    return word


def decode_clip_token(token_id, clip_tokenizer=None):
    """Whole word of a CLIP BPE id, or None if the id is not a whole word that
    re-encodes to exactly this single token."""
    tok = clip_tokenizer or get_clip_tokenizer()
    token = tok.decoder.get(int(token_id))
    if token is None or not token.endswith(CLIP_WORD_END):
        return None  # word-internal fragment or special token
    text = token[:-len(CLIP_WORD_END)]
    try:
        word = bytearray([tok.byte_decoder[c] for c in text]).decode('utf-8')
    except (KeyError, UnicodeDecodeError):
        return None  # partial UTF-8 sequence
    if not word.isalpha():
        return None
    if tok.encode(word) != [int(token_id)]:
        return None
    return word


def embedding_to_words(delta_t, word_table, word_idx, decode, filter_words=None):
    """Map every row of delta_t to the nearest valid vocabulary word.

    delta_t    : (K_p, d) text UAP
    word_table : (V, d) token-embedding table of the source model
    word_idx   : 1-D array of candidate token ids (family vocabulary filter)
    decode     : token id -> word or None (decode_bert_token / decode_clip_token)
    Returns (words, token_ids), one entry per row; rows may map to the same word.
    """
    if filter_words is None:
        filter_words = get_filter_words()
    delta_t = torch.as_tensor(delta_t).detach().float().cpu()
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(0)
    word_table = torch.as_tensor(word_table).detach().float().cpu()
    ids = torch.as_tensor(word_idx, dtype=torch.long).flatten()
    if delta_t.shape[1] != word_table.shape[1]:
        raise ValueError(f'delta_t has dimension {delta_t.shape[1]} but the word-embedding table '
                         f'has dimension {word_table.shape[1]}: the table does not belong to the '
                         f'source model')
    if len(ids) == 0 or int(ids.max()) >= word_table.shape[0] or int(ids.min()) < 0:
        raise ValueError('word_idx contains ids outside the word-embedding table')
    candidates = word_table[ids]
    words, token_ids = [], []
    for row in delta_t:
        dist = torch.linalg.vector_norm(candidates - row.unsqueeze(0), dim=1)
        for k in torch.argsort(dist, stable=True).tolist():
            tid = int(ids[k])
            word = decode(tid)
            if word is None or word in filter_words:
                continue
            words.append(word)
            token_ids.append(tid)
            break
        else:
            raise RuntimeError('no valid vocabulary word found for a row of delta_t')
    return words, token_ids


# ==========================================
# Word substitution
# ==========================================

class TextAttacker:
    """Inserts the adversarial words into captions at the source model's most
    important word positions (see the module docstring).

    source_model : name of the source surrogate (selects BERT or CLIP handling)
    tokenizer    : the source model's BERT tokenizer (BERT family; unused for CLIP)
    adv_words    : the words decoded from delta_t (one per row)
    """

    def __init__(self, source_model, tokenizer, adv_words, max_length=MAX_TEXT_LEN,
                 context_length=CLIP_CONTEXT_LENGTH, batch_size=128, filter_words=None):
        self.source_model = source_model
        self.family = model_family(source_model)
        self.tokenizer = tokenizer
        self.adv_words = list(adv_words)
        self.max_length = max_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.filter_words = get_filter_words() if filter_words is None else filter_words
        if self.family == 'clip':
            self.clip_tokenizer = get_clip_tokenizer()
            self.sot_id = self.clip_tokenizer.encoder['<|startoftext|>']
            self.eot_id = self.clip_tokenizer.encoder['<|endoftext|>']
            self.mask_id = self.clip_tokenizer.encoder[CLIP_MASK_TOKEN]
        elif tokenizer is None:
            raise ValueError('a BERT tokenizer is required for a BERT-family source model')

    # ---------------- tokenization helpers ----------------
    def _word_pieces(self, words):
        """Per word: its tokens (BERT word pieces or CLIP BPE ids)."""
        if self.family == 'clip':
            return [self.clip_tokenizer.encode(w) if w else [] for w in words]
        return [self.tokenizer.tokenize(w) if w else [] for w in words]

    def _max_offset(self):
        # Largest offset (index of the word's first token, counted without
        # [CLS] / <|startoftext|>) that still lies inside the encoded caption.
        if self.family == 'clip':
            return self.context_length - 3   # token index offset + 1 <= 75, before <|endoftext|>
        return self.max_length - 2           # token index offset + 1 <= 29 (C-PGC rule)

    def _pack_clip(self, id_lists):
        """<|startoftext|> + ids + <|endoftext|>, truncated like
        clip.tokenize(..., truncate=True) and zero-padded."""
        out = torch.zeros(len(id_lists), self.context_length, dtype=torch.long)
        for i, ids in enumerate(id_lists):
            tokens = [self.sot_id] + list(ids) + [self.eot_id]
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length]
                tokens[-1] = self.eot_id
            out[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return out

    @staticmethod
    def _device(net):
        try:
            return next(net.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device('cpu')

    @torch.no_grad()
    def _features(self, net, items):
        """L2-normalized source text features of a list of captions (BERT) or
        of BPE id lists (CLIP), in chunks of batch_size."""
        device = self._device(net)
        feats = []
        for s in range(0, len(items), self.batch_size):
            chunk = items[s:s + self.batch_size]
            if self.family == 'clip':
                f = net.encode_text(self._pack_clip(chunk).to(device))
            else:
                text_input = self.tokenizer(chunk, padding='max_length', truncation=True,
                                            max_length=self.max_length, return_tensors='pt').to(device)
                f = net.inference_text(text_input)['text_feat']
            feats.append(F.normalize(f.float().flatten(1), dim=-1))
        return torch.cat(feats, dim=0)

    # ---------------- importance ----------------
    @torch.no_grad()
    def importance_scores(self, net, texts):
        """KL importance of every space-separated word of every caption.
        Returns (words, pieces, scores) with one list per caption."""
        words_all, pieces_all, clean_items, masked_items, owner = [], [], [], [], []
        for i, text in enumerate(texts):
            words = text.split(' ')
            pieces = self._word_pieces(words)
            words_all.append(words)
            pieces_all.append(pieces)
            if self.family == 'clip':
                clean_items.append([t for p in pieces for t in p])
                for w in range(len(words)):
                    masked_items.append([t for k, p in enumerate(pieces)
                                         for t in (p if k != w else [self.mask_id])])
            else:
                clean_items.append(text)
                for w in range(len(words)):
                    masked_items.append(' '.join(words[:w] + ['[UNK]'] + words[w + 1:]))
            owner += [i] * len(words)
        clean = self._features(net, clean_items)
        masked = self._features(net, masked_items)
        owner = torch.as_tensor(owner, dtype=torch.long, device=clean.device)
        kl = F.kl_div(masked.log_softmax(dim=-1), clean.softmax(dim=-1)[owner],
                      reduction='none').sum(dim=-1).cpu().tolist()
        scores, k = [], 0
        for words in words_all:
            scores.append(kl[k:k + len(words)])
            k += len(words)
        return words_all, pieces_all, scores

    # ---------------- substitution ----------------
    def substitute(self, words, pieces, scores):
        """Replace the most important selectable words by the adversarial words."""
        final_words = copy.deepcopy(words)
        offsets, offset = [], 0
        for p in pieces:
            offsets.append(offset)
            offset += len(p)
        order = sorted(range(len(words)), key=lambda w: scores[w], reverse=True)  # stable
        change = 0
        for w in order:
            if change >= len(self.adv_words):
                break
            if words[w] in self.filter_words or len(pieces[w]) == 0:
                continue
            if offsets[w] > self._max_offset():
                continue
            final_words[w] = self.adv_words[change]
            change += 1
        return ' '.join(final_words)

    def get_adv_text(self, net, texts):
        """Adversarial versions of `texts` (list of str) for the source model `net`."""
        texts = list(texts)
        if not texts:
            return []
        words_all, pieces_all, scores = self.importance_scores(net, texts)
        return [self.substitute(w, p, s) for w, p, s in zip(words_all, pieces_all, scores)]
