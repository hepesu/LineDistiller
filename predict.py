import os
import numpy as np
import cv2
from keras.models import load_model

R = 2 ** 3


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

            im_predict = cv2.resize(im, (im.shape[1] // R * R, im.shape[0] // R * R), interpolation=cv2.INTER_AREA)
            im_predict = np.reshape(im_predict, (1, im_predict.shape[0], im_predict.shape[1], 1))
            # im_predict = ((im_predict/255)*220)/255
            im_predict = im_predict.astype(np.float32) * 0.003383

            result = model.predict(im_predict, batch_size=1)[0]

            im_res = (result - np.mean(result) + 1.) * 255
            im_res = cv2.resize(im_res, (im.shape[1], im.shape[0]))

            cv2.imwrite(os.path.join('./output', name), im_res)


if __name__ == "__main__":
    main()
