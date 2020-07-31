import os
import cv2

SIZES = [1.25, 0.75, 0.5, 0.25]


def main():
    for f in ['contour', 'raw']:
        for i in range(1, len(SIZES) + 1):
            if not os.path.exists('./data/%s/%d' % (f, i)):
                os.mkdir('./data/%s/%d' % (f, i))

        # Please ensure files in data/{contour, raw}/0
        for root, dirs, files in os.walk('./data/%s/0' % f, topdown=False):
            for name in files:
                print(os.path.join(root, name))

                im = cv2.imread(os.path.join(root, name))
                h, w = im.shape[:2]

                for i, s in enumerate(SIZES):
                    # Use interpolation method you like
                    im_rs = cv2.resize(im, (int(w * s), int(h * s)))
                    cv2.imwrite('./data/%s/%d/%s' % (f, i + 1, name), im_rs)


if __name__ == "__main__":
    main()
