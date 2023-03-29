import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from phantominator import (
    ct_shepp_logan,
    ct_modified_shepp_logan_params_2d,
)
from collections.abc import Iterable


# Processing functions
def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, preload_cuda=False):
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose(
            [
                Resize(sidelength),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

        self.preload_cuda = preload_cuda
        if preload_cuda:
            self.mgrid = self.mgrid.cuda()
            self.cuda_dataset = [
                self.transform(self.dataset[i]).cuda() for i in range(len(self.dataset))
            ]
        self.step = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.preload_cuda:
            img = self.cuda_dataset[idx]
        else:
            img = self.transform(self.dataset[idx])

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {"idx": idx, "coords": self.mgrid}
        gt_dict = {"img": img, "step": self.step}
        return in_dict, gt_dict


# 2D CT
class Shepp_2d(Dataset):
    def __init__(self, num_train_samples=8, test_samples=8, key=0, scan_id=0, RES=512):
        super().__init__()
        total_samples = num_train_samples + test_samples
        ct_params = np.array(ct_modified_shepp_logan_params_2d())
        shepps = []
        rng = np.random.RandomState(key)
        for i in range(total_samples):
            i_ct_params = ct_params + rng.normal(size=ct_params.shape) / 20
            shepps.append(np.clip(ct_shepp_logan((RES, RES), E=i_ct_params), 0.0, 1.0))

        if not isinstance(scan_id, Iterable):
            scan_id = [scan_id]
        self.img = [Image.fromarray(0.5 * shepps[id] + 0.5) for id in scan_id]
        self.img_channels = 1

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]


def uniform_zero_pad_img(image, padding, pad_val):
    width, height = image.size

    new_width = width + 2 * padding
    new_height = height + 2 * padding

    result = Image.new(image.mode, (new_width, new_height), pad_val)

    result.paste(image, (padding, padding))

    return result


def zero_pad_mask(
    padding,
    padded_sidelength,
    coord_data,
    pad_val=0,
):
    pad_norm = 2 * padding / padded_sidelength
    coordinates = coord_data[0][0]["coords"]
    coord_mask = torch.ones(coordinates.size())
    coord_mask[(coordinates > 1 - pad_norm)] = pad_val
    coord_mask[(coordinates < -1 + pad_norm)] = pad_val
    coord_mask = coord_mask[:, 0] * coord_mask[:, 1]
    return coord_mask


class AAPM_2d(Dataset):
    def __init__(
        self,
        test_set=False,
        scan_id=0,
        PADDING=0,
        ZERO_PAD_VAL=0.5,
        DATA_PATH="/data/ziz/not-backed-up/bhe/implicit-uncertainty/data",
    ):
        super().__init__()

        # Change to your filesystem.
        if test_set:
            directory_path = f"{DATA_PATH}/AAPM_test_set/"
        else:
            directory_path = f"{DATA_PATH}/AAPM_val_set/"

        aapms = []
        for file in sorted(os.listdir(directory_path)):
            file_path = directory_path + file
            aapms.append(np.load(file_path))

        if not isinstance(scan_id, Iterable):
            scan_id = [scan_id]

        self.img = [
            uniform_zero_pad_img(
                Image.fromarray(0.5 * aapms[id] + 0.5), PADDING, ZERO_PAD_VAL
            )
            for id in scan_id
        ]

        self.img_channels = 1

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]
