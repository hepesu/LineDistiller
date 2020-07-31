import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as F
import PIL


def filter(input):
    return input.filter(PIL.ImageFilter.GaussianBlur(1))


def transform(input, target, size):
    iw, ih = input.size
    a, t, sc, sh = transforms.RandomAffine.get_params((0, 180), (0.3, 0.3), (0.7, 1.4), (-5, 5), (iw, ih))
    jitter = transforms.ColorJitter.get_params((0.1, 1.2), (0.8, 1.2), (0.0, 1.2), (-0.1, 0.1))

    target = F.to_grayscale(target)

    if random.random() > 0.5:
        input = F.vflip(input)
        target = F.vflip(target)

    if random.random() > 0.5:
        input = F.hflip(input)
        target = F.hflip(target)

    input = F.affine(input, a, t, sc, sh, fillcolor=(0, 0, 0), resample=2)
    target = F.affine(target, a, t, sc, sh, fillcolor=255, resample=2)

    input_full = F.resize(input, (size, size))

    input = F.center_crop(input, (size, size))
    target = F.center_crop(target, (size, size))

    input = jitter(input)
    input_full = jitter(input_full)
    input_filtered = filter(input)

    input = F.to_tensor(input)
    input_full = F.to_tensor(input_full)
    input_filtered = F.to_tensor(input_filtered)
    target = F.to_tensor(target)

    input_filtered = input - input_filtered + 0.5

    input = F.normalize(input, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    input_full = F.normalize(input_full, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    input_filtered = F.normalize(input_filtered, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    target = F.normalize(target, [0.5], [0.5])

    return input, target, input_full, input_filtered


class PairDataset(Dataset):
    def __init__(self, rawDataset, contourDataset, size=128):
        self.rawDataset = rawDataset
        self.contourDataset = contourDataset
        self.size = size

        assert (len(self.rawDataset) == len(self.contourDataset)), 'Datasets should have equal size'

    def __len__(self):
        return len(self.rawDataset)

    def __getitem__(self, idx):
        input = self.rawDataset[idx][0]
        target = self.contourDataset[idx][0]

        return transform(input, target, self.size)


if __name__ == '__main__':
    import cv2
    import numpy as np

    pairDataLoader = DataLoader(
        PairDataset(
            datasets.ImageFolder(root='./data/raw'),
            datasets.ImageFolder(root='./data/contour')
        ),
        batch_size=1,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    pairDataLoaderIter = iter(pairDataLoader)
    rawData, contourData, rawFullData, rawFilteredData = next(pairDataLoaderIter)

    raw = ((rawData[0] + 1) / 2 * 255).permute(1, 2, 0).numpy()
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./raw_check.png', raw)

    contour = ((contourData[0] + 1) / 2 * 255).permute(1, 2, 0).numpy()
    cv2.imwrite('./contour_check.png', contour)

    rawFull = ((rawFullData[0] + 1) / 2 * 255).permute(1, 2, 0).numpy()
    rawFull = cv2.cvtColor(rawFull, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./rawFull_check.png', rawFull)

    rawFiltered = ((rawFilteredData[0] + 1) / 2 * 255).permute(1, 2, 0).numpy()
    rawFiltered = cv2.cvtColor(rawFiltered, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./rawFiltered_check.png', rawFiltered)
