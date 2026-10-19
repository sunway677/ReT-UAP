# Third-party notices

The ReT-UAP-specific code (`retuap_train.py`, `retuap_targets.py`,
`retuap_attacker.py`, `retuap_generator.py` except the parts credited below,
`eval.py` and `scripts/`) is released under the MIT license in
[`LICENSE`](LICENSE). This repository also redistributes code and data from the
projects below, which remain under their original licenses. Full license texts
are in [`licenses/`](licenses/); the original copyright headers are kept in the
individual files where the upstream file has one.

Modified third-party files: the text-perturbation insertion and the code
around it are changed for ReT-UAP in `models/xbert.py` and
`models/model_retrieval.py` (ALBEF/TCL), `models/blip_model/med.py` and
`models/blip_model/blip_retrieval.py` (BLIP), `models/xvlm_model/xbert.py` and
`models/xvlm_model/model_retrieval.py` (X-VLM), and `models/clip_model/model.py`
and `models/clip_model/clip.py` (CLIP). The changes are marked
"modified for ReT-UAP" in the source; these files remain under their upstream
licenses. `models/tokenization_bert.py` loads its vocabulary before calling
`super().__init__` (a change inherited from the C-PGC code base).

| Component | Files in this repo | License | Copyright |
|---|---|---|---|
| [ALBEF](https://github.com/salesforce/ALBEF) (also the architecture of the TCL checkpoints) | `models/model_retrieval.py` (modified), `models/xbert.py` (modified), `models/vit.py`, `models/tokenization_bert.py`, `configs/config_bert.json` | BSD-3-Clause ([text](licenses/ALBEF_LICENSE.txt)) | (c) 2021, Salesforce.com, Inc. |
| [BLIP](https://github.com/salesforce/BLIP) | `models/blip_model/{blip,blip_retrieval,med,vit}.py`, `models/blip_model/med_config.json` (`blip_retrieval.py` and `med.py` modified) | BSD-3-Clause ([text](licenses/BLIP_LICENSE.txt)) | (c) 2022, Salesforce.com, Inc. |
| [X-VLM](https://github.com/zengyan-97/X-VLM) | `models/xvlm_model/` (`xbert.py` and `model_retrieval.py` modified) | BSD-3-Clause ([text](licenses/X-VLM_LICENSE.txt)) | (c) 2022, ByteDance Inc. |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (via X-VLM) | `models/xvlm_model/swin_transformer.py` | MIT ([text](licenses/Swin-Transformer_LICENSE.txt)) | (c) Microsoft Corporation |
| [CLIP](https://github.com/openai/CLIP) | `models/clip_model/` (`clip.py` and `model.py` modified), including the BPE vocabulary `bpe_simple_vocab_16e6.txt.gz` from which `word_idx_CLIP.npy` is derived | MIT ([text](licenses/CLIP_LICENSE.txt)) | (c) 2021 OpenAI |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) (BERT / RoBERTa / CLIP-ViT code adapted by ALBEF, BLIP and X-VLM) | `models/xbert.py` (modified), `models/tokenization_bert.py`, `models/blip_model/med.py` (modified), `models/xvlm_model/{xbert,xroberta,clip_vit}.py` (`xbert.py` modified) | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) The Google AI Language Team Authors, The HuggingFace Inc. team, NVIDIA Corporation, The OpenAI Team Authors |
| [timm](https://github.com/huggingface/pytorch-image-models) (ViT code adapted by ALBEF, BLIP and X-VLM) | `models/vit.py`, `models/blip_model/vit.py`, `models/xvlm_model/vit.py` | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) 2019 Ross Wightman |
| [DETR](https://github.com/facebookresearch/detr) (via X-VLM) | `models/xvlm_model/box_ops.py` | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) Facebook, Inc. and its affiliates |
| [latent-diffusion](https://github.com/CompVis/latent-diffusion) (via C-PGC's `attention.py`) | the design of the cross-attention block in `retuap_generator.py` | MIT ([text](licenses/latent-diffusion_LICENSE.txt)) | (c) 2022 Machine Vision and Learning Group, LMU Munich |
| [C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks) (Fang et al., ICCV 2025) | `attacker.py` (caption word substitution, derived from C-PGC's `TextAttacker`), `utils.py`, `dataset.py`, `configs/`, the annotation files in `data_annotation/` as distributed with C-PGC, the deconvolution generator backbone in `retuap_generator.py`, and the base structure of the evaluation protocol in `eval.py`; `word_idx_BERT.npy` is the BERT vocabulary filter read by the C-PGC evaluation code (not included in the C-PGC repository) | no license file in the upstream repository | (c) the C-PGC authors |
| Flickr30K captions (as distributed with C-PGC, ALBEF annotation format) | `data_annotation/flickr30k_{train,test}.json` | [Flickr30K terms of use](https://shannon.cs.illinois.edu/DenotationGraph/) (non-commercial research and educational use) | (c) the Flickr30K authors |
| MS-COCO captions (as distributed with C-PGC, ALBEF annotation format) | `data_annotation/coco_{train,test}.json` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([COCO terms of use](https://cocodataset.org/#termsofuse)) | (c) the COCO Consortium |

The Flickr30K and MS-COCO images (subject to the
[Flickr terms of use](https://www.flickr.com/creativecommons/)), the
fine-tuned ALBEF, TCL, BLIP and X-VLM checkpoints, the `bert-base-uncased`
weights and the CLIP weights are not redistributed; download them from the
original sources listed in the README.
