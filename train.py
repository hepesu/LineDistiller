import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import datasets

from model import Net
from dataset import PairDataset

SIZE = 256
ITERATIONS = 8000
BATCH_SIZE = 8
SEED = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------
    #  Model and Optimizer
    # ----------
    model = Net(in_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.5, last_epoch=-1)

    # ----------
    #  Data
    # ----------
    pairDataLoader = torch.utils.data.DataLoader(
        PairDataset(
            datasets.ImageFolder(root='./data/raw'),
            datasets.ImageFolder(root='./data/contour'),
            size=SIZE
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    pairDataLoaderIter = iter(pairDataLoader)

    # ----------
    #  Training
    # ----------
    for iteration in range(1, ITERATIONS + 1):
        try:
            rawData, contourData, rawFullData, rawFilteredData = next(pairDataLoaderIter)
        except StopIteration:
            pairDataLoaderIter = iter(pairDataLoader)
            rawData, contourData, rawFullData, rawFilteredData = next(pairDataLoaderIter)

        rawData = rawData.to(DEVICE)
        contourData = contourData.to(DEVICE)

        optimizer.zero_grad()

        loss = F.l1_loss(model(rawData), contourData)

        loss.backward()
        optimizer.step()

        print("[Iteration %d] [loss: %f]" % (iteration, loss.item()))

        if iteration % 100 == 0:
            torch.save(model.state_dict(), os.path.join('weight', '{}.pth'.format(iteration)))

        if iteration < 4000:
            scheduler.step()


if __name__ == "__main__":
    main()
