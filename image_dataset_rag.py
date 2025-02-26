import os
import json
import torch
import functools
import numpy as np
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self,
                 meta_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 set,
                 test,
                 get_loader=get_default_img_loader
    ):
        with open(meta_file) as fr:
            self.metas = json.load(fr)
        print('%d meta data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):
        eps = 1e-2
        sample_A = self.get_oneitem(index)
        while True:
            index_B = random.randint(0, len(self) - 1)
            sample_B = self.get_oneitem(index_B)
            if abs(sample_A["mos"] - sample_B["mos"]) > eps:
                break
        sample = {
            "I_A": sample_A["I"],
            "text_A": sample_A["text"],
            "mos_A": sample_A["mos"],
            "I_B": sample_B["I"],
            "text_B": sample_B["text"],
            "mos_B": sample_B["mos"],
        }
        return sample

    def get_oneitem(self, index):
        meta = self.metas[index]
        image_name = os.path.join(self.img_dir, meta["image"])
        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)
        assert patches.size(0) >= self.num_patch, f"{patches.size(0)} {self.num_patch} {I.shape}"
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]

        text = meta["conversations"][1]["value"]
        mos = meta["gt_score_norm"]
        sample = {
            "I": patches,
            "text": text,
            "mos": float(mos),
        }
        return sample

    def __len__(self):
        return len(self.metas)
