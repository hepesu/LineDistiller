import numpy as np
import cv2
from .fill import fill, fill_ave_mask

KERNEL = np.ones((5, 5), np.uint8)


def simple_mask(im_lines):
    """
    Compute polygon, bounding rect and rough mask based on lines from image.

    :return: poly, rect, mask
    """
    _, thresh = cv2.threshold(255 - im_lines, 10, 255, 0)

    # Get overall poly mask
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask0 = np.zeros_like(im_lines)

    for ctn in contours:
        pts = cv2.approxPolyDP(ctn, 50, True)

        if len(pts) > 2:
            convex_pts = cv2.convexHull(pts)
            mask0 = cv2.fillPoly(mask0, [convex_pts], 255)

    # Get overall bounding rect
    _, contours, _ = cv2.findContours(mask0, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    poly = contours
    rect = [cv2.boundingRect(c) for c in contours]

    return poly, rect, mask0


def mask(im_color, im_lines, use_grabcut=True):
    """
    Compute mask based on lines from image, then refine it with GrabCut

    :return: poly, rect, mask_fill, mask_grab
    """
    poly, rect, mask0 = simple_mask(im_lines)

    # Get mask with fillmap
    fillmap = fill(im_lines)

    mask0_expanded = cv2.dilate(mask0, KERNEL, iterations=20)
    mask0_averaged = fill_ave_mask(mask0_expanded, fillmap)
    mask0_shrinked = cv2.erode(mask0_averaged, KERNEL, iterations=10)

    mask_res = np.where(mask0_averaged == 255, 1, 0)

    # Get refined mask with GrabCut(other matting methods also work)
    if use_grabcut and not np.average(mask0_expanded) > 254 and not np.average(mask0_shrinked) > 254:
        # Build mask for GrabCut
        mask0_gc = np.zeros_like(im_lines)
        mask0_gc[np.where(mask0_expanded == 255)] = 2
        mask0_gc[np.where(mask0_shrinked == 255)] = 1

        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        mask_refined, bgModel, fgModel = cv2.grabCut(im_color, mask0_gc, None, bgModel, fgModel, iterCount=2,
                                                     mode=cv2.GC_INIT_WITH_MASK)

        mask_res = np.zeros_like(im_lines)
        mask_res[np.where((mask_refined == 3) | (mask_refined == 1))] = 255

    return poly, rect, mask0_averaged, mask_res


if __name__ == '__main__':
    filename = 'sample'
    im_color = cv2.imread('../input/%s.png' % filename)
    im_lines = cv2.imread('../output/%s.png' % filename, cv2.IMREAD_GRAYSCALE)

    poly, rect, mask_fill, mask_grab = mask(im_color, im_lines)

    mask_poly = np.zeros_like(im_color)
    for i in range(len(poly)):
        cv2.fillPoly(mask_poly, [poly[i]], (255, 255, 255))
        cv2.drawContours(mask_poly, [poly[i]], 0, (255, 0, 0), 5)
        cv2.rectangle(mask_poly, (rect[i][0], rect[i][1]), (rect[i][0] + rect[i][2], rect[i][1] + rect[i][3]),
                      (0, 0, 255), 5)

    cv2.imwrite('./result_mask_poly.png', mask_poly)
    cv2.imwrite('./result_mask_fill.png', mask_fill)
    cv2.imwrite('./result_mask_grab.png', mask_grab)
    cv2.imwrite('./result_mask_out.png', im_color * (mask_grab[:, :, np.newaxis] / 255))
