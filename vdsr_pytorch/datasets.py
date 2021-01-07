import os
import numpy as np

import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def check_image_file(filename):
    r"""Determine whether the files in the directory are in image format.
    Args:
        filename (str): The current path of the image
    Returns:
        Returns True if it is an image and False if it is not.
    """
    return any(filename.endswith(extension) for extension in [".bmp", ".BMP",
                                                              ".jpg", ".JPG",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])


def cutblur(hr_img, lr_img, prob=0.5, alpha=0.7):

    if hr_img.size() != lr_img.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return hr_img, lr_img

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = lr_img.size(1), lr_img.size(2)
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
    cy = np.random.randint(0, h - ch + 1)
    cx = np.random.randint(0, w - cw + 1)

    # apply CutBlur to inside or outside

    if np.random.random() > 0.5:
        lr_img[..., cy:cy + ch, cx:cx + cw] = hr_img[..., cy:cy + ch, cx:cx + cw]
    else:
        lr_img_aug = hr_img.clone()
        lr_img_aug[..., cy:cy + ch, cx:cx + cw] = lr_img[..., cy:cy + ch, cx:cx + cw]
        lr_img = lr_img_aug

    return hr_img, lr_img


def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    return im1, im2


def blend(im1, im2, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(1), im2.size(2)))
    rim1 = c.repeat((1, 1, im1.size(1), im1.size(2)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1-v) * rim1
    im2 = v * im2 + (1-v) * rim2

    return im1, im2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2


def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(1), im2.size(2)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2


def cutmixup(
    im1, im2,
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def cutout(im1, im2, prob=1.0, alpha=0.1):
    scale = im1.size(2) // im2.size(2)
    fsize = (im2.size(0), 1)+im2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fim2 = np.ones(fsize)
        fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
        fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")
        return im1, im2, fim1, fim2

    fim2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
    fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")

    im2 *= fim2

    return im1, im2, fim1, fim2


class DatasetFromFolder(Dataset):
    def __init__(self, images_dir, image_size=256, scale_factor=4, transform=None):
        r""" Dataset loading base class.
        Args:
            images_dir (str): The directory address where the image is stored.
            image_size (int): Original high resolution image size. Default: 256.
            scale_factor (int): Coefficient of image scale. Default: 4.
        """
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(images_dir, x) for x in
                                os.listdir(images_dir)
                                if check_image_file(x)]
        self.transform = transform

        crop_size = image_size - (image_size % scale_factor)  # Valid crop size
        self.input_transform = transforms.Compose(
            [transforms.CenterCrop(crop_size),  # cropping the image
             transforms.RandomChoice([
                 # mimic the lr images and upscale it back
                 transforms.Resize(crop_size // 3, interpolation=0),
                 transforms.GaussianBlur(3),
             ]),
             transforms.RandomChoice([
                 transforms.Resize(crop_size, interpolation=2),
             ]),
             transforms.ToTensor()])
        self.target_transform = transforms.Compose(
            [transforms.CenterCrop(crop_size),
             # since it's the target, we keep its original quality
             transforms.ToTensor()])

    def __getitem__(self, index):
        r""" Get image source file
        Args:
            index (int): Index position in image list.
        Returns:
            Low resolution image and high resolution image.
        """
        # Do on Y channel (1 in total) or RGB channels (3 in total)

        #image = Image.open(self.image_filenames[index]).convert("YCbCr")
        image = Image.open(self.image_filenames[index]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        #inputs, _, _ = image.split()
        inputs = image
        target = inputs.copy()

        inputs = self.input_transform(inputs)
        target = self.target_transform(target)

        #target, inputs = blend(target, inputs, 0.4, 0.6)
        target, inputs = mixup(target, inputs, 0.4, 1.2)
        target, inputs, _ , _= cutout(target, inputs, 0.4, 0.001)
        target, inputs = cutblur(target, inputs, 0.4, 0.7)
        #target, inputs = rgb(target, inputs, 0.4)

        return inputs, target

    def __len__(self):
        return len(self.image_filenames)