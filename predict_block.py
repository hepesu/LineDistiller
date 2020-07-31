import os
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model

SIZE = 128
BATCH_SIZE = 4
PAD_SIZE = 8

K.set_learning_phase(0)


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


def preprocess(x):
    return np.reshape(x, (1, x.shape[0], x.shape[1], 1)).astype(np.float32) * 0.003383


def main():
    model = load_model('model.h5')

    for root, dirs, files in os.walk('./input', topdown=False):
        for name in files:
            print(os.path.join(root, name))

            im = cv2.imread(os.path.join(root, name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # perform brightness correction in tiles
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            im = clahe.apply(im)

            im_crops = to_crop(im, SIZE)

            im_crops_res = []
            for c in range(0, im_crops.shape[0], BATCH_SIZE):
                batch = np.concatenate([preprocess(im_crop) for im_crop in im_crops[c:c + BATCH_SIZE]], 0)

                res = model.predict_on_batch(batch)

                for r in res:
                    im_crops_res.append((r - np.mean(r) + 1.) * 255)

            im_res = from_crop(im, im_crops_res, SIZE)
            cv2.imwrite(os.path.join('./output', name), im_res)


if __name__ == "__main__":
    main()
