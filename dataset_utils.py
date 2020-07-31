import os
import numpy as np
import cv2
from utils.mask import simple_mask, mask
from utils.fill import fill, fill_ave
from utils.misc import shade_color

VALID_RATIO = 0.2


def main():
    for root, dirs, files in os.walk('./input', topdown=False):
        for name in files:
            print(os.path.join(root, name))

            im_color = cv2.imread('input/%s' % name)
            im_lines = cv2.imread('output/%s' % name, cv2.IMREAD_GRAYSCALE)
            h, w = im_lines.shape[:2]

            poly, rect, mask_poly = simple_mask(im_lines)

            # Skip invalid
            if not np.sum(mask_poly) / (255 * h * w) > VALID_RATIO:
                continue

            # Massive computation work start
            poly, rect, mask_fill, mask_grab = mask(im_color, im_lines)

            # If you need full fillmap, use_floodfill = True
            fillmap = fill(im_lines, use_floodfill=False)

            # Output
            cv2.imwrite('./dataset/raw/%s' % name, im_color)
            cv2.imwrite('./dataset/contour/%s' % name, im_lines)

            # Output mask
            cv2.imwrite('./dataset/mask/%s' % name, mask_grab)
            cv2.imwrite('./dataset/mask/p_%s' % name, mask_fill)

            # Output fill
            np.save('./dataset/fill/%s' % name.replace('png', 'npy'), fillmap)
            cv2.imwrite('./dataset/flat/%s' % name, fill_ave(im_color, fillmap, ave_line=False))


if __name__ == "__main__":
    main()
