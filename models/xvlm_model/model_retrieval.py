import torch
from models.xvlm_model import XVLMBase, load_pretrained
import torch.nn.functional as F

class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx=None):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        return loss_itc, loss_itm
    
    def inference_text(self, text_input):
        text_embed = self.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)
        return {'text_feat': text_feat, 'text_embed': text_embed}
    
    def inference_text_replace(self, text_input, embedding, position, padding_zero):
        # modified for ReT-UAP: validate the substitution positions and shorten a
        # copy of the attention mask (text_input is no longer edited in place). The
        # vendored BertTokenizer encodes a caption as [CLS] + word pieces with no
        # trailing [SEP], so the caption's content ends at the last attended token; a
        # final [SEP] is excluded only if the tokenizer appended one. A substitution
        # at token index t = offset + 1 is kept only if t lies inside the content, its
        # extra word pieces are clipped to the content (a word cut by truncation), and
        # the mask loses as many trailing tokens as the embedding layer removes.
        sep_id = getattr(getattr(self, 'tokenizer', None), 'sep_token_id', None)
        attention_mask = text_input.attention_mask.clone()
        valid_position, valid_padding_zero = [], []
        for i in range(len(position)):
            n = int(attention_mask[i].sum().item())
            end = n  # exclusive end of the caption's content
            if sep_id is not None and int(text_input.input_ids[i, n - 1]) == sep_id:
                end = n - 1
            row_position, row_padding_zero = [], []
            for offset, n_extra in zip(position[i], padding_zero[i]):
                t = None if offset is None else offset + 1
                if t is None or n_extra < 0 or t < 1 or t > end - 1:
                    row_position.append(None)
                    row_padding_zero.append(0)
                    continue
                row_position.append(offset)
                row_padding_zero.append(min(n_extra, end - 1 - t))
            removed = sum(row_padding_zero)
            if removed > 0:
                attention_mask[i, n - removed:] = 0
            valid_position.append(row_position)
            valid_padding_zero.append(row_padding_zero)
        text_embed = self.get_text_embeds(text_input.input_ids, attention_mask, replace_embedding=embedding,
                                          replace_position=valid_position, padding_zero=valid_padding_zero)
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)
        return {'text_feat': text_feat, 'text_embed': text_embed}

    def inference_image(self, image):
        image_embed, _ = self.get_vision_embeds(image)
        image_feat = F.normalize(self.vision_proj(image_embed[:, 0, :]), dim=-1)
        return {'image_feat': image_feat, 'image_embed': image_embed}