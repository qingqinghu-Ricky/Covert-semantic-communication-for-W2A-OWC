"""Source training augmentations and evaluation resize, including historical BGR order."""
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ToTensor(object):
    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomRotation(object):
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            h, w = image.shape[:2]
            angle = random.uniform(-self.degrees, self.degrees)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomBrightnessContrast(object):
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            image = image.astype(np.float32)
            brightness_factor = random.uniform(*self.brightness_range)
            image = image * brightness_factor
            contrast_factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean
            image = np.clip(image, 0, 255).astype(np.uint8)
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomGammaCorrection(object):
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            inv_gamma = 1.0 / gamma
            image = (image / 255.0) ** inv_gamma * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomTranslate(object):
    def __init__(self, translate_range=(0.1, 0.1), p=0.5):
        self.x_range = translate_range[0]
        self.y_range = translate_range[1]
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            h, w = image.shape[:2]
            tx = int(random.uniform(-self.x_range, self.x_range) * w)
            ty = int(random.uniform(-self.y_range, self.y_range) * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomShear(object):
    def __init__(self, shear_range=0.2, p=0.5):
        self.shear_range = shear_range
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            h, w = image.shape[:2]
            shear_factor = random.uniform(-self.shear_range, self.shear_range)
            M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            tx = -shear_factor * h / 2
            M[0, 2] = tx
            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            mask = cv2.warpAffine(
                mask,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return {"image": image, "mask": mask, "name": sample["name"]}


class RandomScale(object):
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, sample):
        image, mask = (sample["image"], sample["mask"])
        if random.random() < self.p:
            h, w = image.shape[:2]
            scale = random.uniform(*self.scale_range)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_scaled = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            mask_scaled = cv2.resize(
                mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
            if scale > 1.0:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = image_scaled[start_y : start_y + h, start_x : start_x + w]
                mask = mask_scaled[start_y : start_y + h, start_x : start_x + w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                pad_x2 = w - new_w - pad_x
                pad_y2 = h - new_h - pad_y
                image = cv2.copyMakeBorder(
                    image_scaled,
                    pad_y,
                    pad_y2,
                    pad_x,
                    pad_x2,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
                mask = cv2.copyMakeBorder(
                    mask_scaled,
                    pad_y,
                    pad_y2,
                    pad_x,
                    pad_x2,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        return {"image": image, "mask": mask, "name": sample["name"]}


class ResizeWithPaddingCV(object):
    def __init__(self, target_size, image_pad_value=(0, 0, 0), mask_pad_value=0):
        if isinstance(target_size, int):
            self.target_w, self.target_h = (target_size, target_size)
        elif isinstance(target_size, tuple) and len(target_size) == 2:
            self.target_w, self.target_h = target_size
        else:
            raise ValueError("target_size必须是int或长度为2的tuple（width, height）")
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value

    def _process(self, img, is_mask=False):
        img_cv = img
        h, w = img_cv.shape[:2]
        c = img_cv.shape[2] if len(img_cv.shape) == 3 else 1
        scale_w = self.target_w / w
        scale_h = self.target_h / h
        scale = min(scale_w, scale_h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if is_mask:
            img_scaled = cv2.resize(
                img_cv, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            img_scaled = cv2.resize(
                img_cv, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        pad_top = (self.target_h - new_h) // 2
        pad_bottom = self.target_h - new_h - pad_top
        pad_left = (self.target_w - new_w) // 2
        pad_right = self.target_w - new_w - pad_left
        pad_value = self.mask_pad_value if is_mask else self.image_pad_value
        if isinstance(pad_value, int):
            pad_value = (pad_value,) * c
        padded_img = cv2.copyMakeBorder(
            img_scaled,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
        if not is_mask:
            padded_img_pil = Image.fromarray(
                cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
            )
        else:
            padded_img_pil = Image.fromarray(padded_img)
        return padded_img_pil

    def __call__(self, sample):
        image_padded = self._process(sample["image"], is_mask=False)
        mask_padded = self._process(sample["mask"], is_mask=True)
        return {"image": image_padded, "mask": mask_padded, "name": sample["name"]}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
