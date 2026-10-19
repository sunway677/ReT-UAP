import json 
import os
import re

import torch 
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset


def image_key(image_path):
    """Identifier of an image across annotation files (its file name; the Flickr30K
    and MS-COCO file names are unique)."""
    return os.path.basename(image_path)


def load_image_keys(ann_file):
    """Set of image identifiers (see image_key) listed in an annotation file."""
    return {image_key(ann['image']) for ann in json.load(open(ann_file, 'r'))}


def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class paired_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, exclude_images=None):
        self.ann = json.load(open(ann_file, 'r'))
        # Added for ReT-UAP: drop every entry whose image is in `exclude_images`
        # (a set of image_key identifiers). The training script passes the test
        # images so that the attacker never sees test data (threat model); the
        # shipped Flickr30K training annotation, inherited from C-PGC, contains
        # 955 of the 1000 Flickr30K test images.
        self.num_excluded = 0
        if exclude_images:
            kept = [ann for ann in self.ann if image_key(ann['image']) not in exclude_images]
            self.num_excluded = len(self.ann) - len(kept)
            self.ann = kept
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            self.image.append(ann['image'])
            for j, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text_ids =  self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups