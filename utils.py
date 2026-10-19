"""
Helpers shared by retuap_train.py, eval.py and attacker.py (adapted from C-PGC):

- load_model: builds one of the six surrogate / target models
  (ALBEF, TCL, BLIP, XVLM, ViT-B/16, ViT-B/32) from its fine-tuned checkpoint
  (CLIP weights are loaded by name).
- get_filter_words: stop words and punctuation that are never used as, or
  replaced by, adversarial words.

Relative paths (the X-VLM and BLIP configuration files) are resolved against
the working directory, so run the code from the repository root.
"""
import os

import torch

from models.blip_model.blip_retrieval import BLIP_Retrieval
from models.xvlm_model.model_retrieval import XVLM
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip
from transformers import BertForMaskedLM
from ruamel.yaml import YAML


# ==========================================
# Model loading
# ==========================================

CLIP_MODELS = ('ViT-B/16', 'ViT-B/32')


def load_model(model_name, model_ckpt, text_encoder, config, device):
    """
    Load a retrieval model by name and return (model, ref_model, tokenizer).

    Supported model_name values:
      'ALBEF', 'TCL'         ALBEF architecture (TCL checkpoints use it too)
      'BLIP'                 BLIP retrieval model
      'XVLM'                 X-VLM retrieval model
      'ViT-B/16', 'ViT-B/32' OpenAI CLIP (loaded with clip.load; model_ckpt is ignored)

    model_ckpt is the fine-tuned checkpoint (<checkpoint_dir>/<MODEL>/<DATASET>.pth),
    text_encoder the Hugging Face name of the BERT model ('bert-base-uncased'),
    config the dataset YAML config (its 'dataset' entry selects the X-VLM config).
    tokenizer is the BERT tokenizer of the model (with [DEC]/[ENC] added for BLIP;
    for CLIP it is attached to the model with set_tokenizer).
    ref_model is a BertForMaskedLM built from text_encoder; it is kept for
    interface compatibility with C-PGC (callers of this release discard it).
    The model is returned on the CPU for ALBEF/TCL/BLIP/XVLM and on `device`
    for CLIP; callers move it to the target device.
    """
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)
    tokenizer = BertTokenizer.from_pretrained(text_encoder)

    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location='cpu')

    # CLIP models are loaded by name; return early (no checkpoint to load).
    elif model_name in CLIP_MODELS:
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer

    elif model_name == 'XVLM':
        xvlm_config_path = os.path.join('models', 'xvlm_model',
                                        'Retrieval_{}.yaml'.format(config['dataset']))
        with open(xvlm_config_path, 'r') as f:
            xvlm_config = YAML(typ='safe', pure=True).load(f)
        model = XVLM(config=xvlm_config)
        model.load_pretrained(model_ckpt, xvlm_config, is_eval=True)
        return model, ref_model, tokenizer

    elif model_name == 'BLIP':
        # BLIP requires special decoder/encoder tokens to be registered first.
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        model = BLIP_Retrieval()
        checkpoint = torch.load(model_ckpt, map_location='cpu')

    else:
        raise ValueError('unknown model {!r}; expected one of {}'.format(
            model_name, ('ALBEF', 'TCL', 'BLIP', 'XVLM') + CLIP_MODELS))

    # Support both {'model': state_dict} and bare state_dict checkpoint formats.
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if model_name in ['TCL', 'BLIP']:
        # Interpolate positional embeddings to match the model's image resolution.
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

    # Strip the 'bert.' prefix that some checkpoints add to text encoder keys.
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]

    if model_name == 'BLIP':
        # Drop any keys whose tensor shapes do not match the current model.
        for key in model.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model, ref_model, tokenizer


# ==========================================
# Stop-word filter
# ==========================================

def get_filter_words():
    """
    Return a set of common stop words and punctuation that should not be
    used as adversarial word substitutions.
    """
    filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
    filter_words = set(filter_words)
    return filter_words
