import os
import random
import numpy as np
import cv2

import torch
from torchvision import transforms

from model import Net

import PIL

SIZE = 128
BATCH_SIZE = 4
PAD_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_crop(im, size):
    height, width = im.shape[:2]

    pad_height = size * int(np.ceil(height / float(size))) - height
    pad_width = size * int(np.ceil(width / float(size))) - width

    im_pad = cv2.copyMakeBorder(im, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT)

    im_crops = []
    for i in range(0, height, size):
        for j in range(0, width, size):
            im_crop = im_pad[i:i + size, j:j + size]
            im_crops.append(cv2.copyMakeBorder(im_crop, PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE, cv2.BORDER_REFLECT))

    return np.array(im_crops)


def from_crop(im, im_crops, size):
    height, width = im.shape[:2]

    im_pad = np.zeros((
        size * int(np.ceil(height / float(size))),
        size * int(np.ceil(width / float(size))),
        1
    ))

    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            im_pad[i:i + size, j:j + size] = im_crops[idx][PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE]
            idx += 1

    return im_pad[:height, :width, :]


def main():
    model = Net(in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])

    for root, dirs, files in os.walk('./input', topdown=False):
        for name in files:
            print(os.path.join(root, name))

            im = cv2.imread(os.path.join(root, name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            im_crops = to_crop(im, SIZE)

            im_crops_res = []
            for c in range(0, im_crops.shape[0], BATCH_SIZE):
                batch_im = torch.cat([preprocess(im_crop).unsqueeze(0) for im_crop in im_crops[c:c + BATCH_SIZE]], 0)

                batch = batch_im.to(DEVICE)

                res = model(batch).permute(0, 2, 3, 1).detach().cpu().numpy()

                for r in res:
                    im_crops_res.append((r + 1) / 2 * 255)

            im_res = from_crop(im, im_crops_res, SIZE)
            cv2.imwrite(os.path.join('./output', name), im_res)


if __name__ == "__main__":
    main()
