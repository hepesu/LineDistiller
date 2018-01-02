import os
import numpy as np
import cv2
from keras.models import load_model

IMG_WIDTH, IMG_HEIGHT = 1280,720
MODEL_NAME = 'model.h5'

model = load_model(MODEL_NAME)

for root, dirs, files in os.walk("data/predict", topdown=False):
    for name in files:
        print(os.path.join(root, name))

        im = cv2.imread(os.path.join(root, name))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # perform brightness correction in titles
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        im_gray = clahe.apply(im_gray)

        im_predict = cv2.resize(im_gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        im_predict = np.reshape(im_predict, (1, im_predict.shape[0], im_predict.shape[1], 1))
        im_predict = im_predict.astype(np.float32) / 255.

        result = model.predict(im_predict, batch_size=1)

        im_res = np.reshape(result, (im_predict.shape[1], im_predict.shape[2]))
        cv2.imwrite(os.path.join('data/result', name), (im_res - np.mean(im_res) + 1.) * 255)
