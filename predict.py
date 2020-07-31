import os
import random
import numpy as np
import cv2

import torch
from torchvision import transforms

from model import Net

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R = 2 ** 3


def main():
    model = Net().to(DEVICE)
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
            im_resized = cv2.resize(im, (im.shape[1] // R * R, im.shape[0] // R * R))

            res = model(preprocess(im_resized).unsqueeze(0))

            im_res = (res.squeeze(0).permute(1, 2, 0).detach().numpy() + 1) / 2 * 255
            im_res = cv2.resize(im_res, (im.shape[1], im.shape[0]))

            cv2.imwrite(os.path.join('./output', name), im_res)


if __name__ == "__main__":
    main()
