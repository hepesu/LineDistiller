import numpy as np
import cv2
from .linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, \
    merge_fill, show_fill_map


def fill(im, use_floodfill=False):
    """
    Fill lines with trapped-ball fill and floodfill.

    Please refer to LineFiller for more information:
    https://github.com/hepesu/LineFiller/

    ** Warning ** Save fillmap as image resulting truncation at maximum value 255

    :return: fillmap
    """
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary

    _fill = trapped_ball_fill_multi(result, 5, method='max')
    result = mark_fill(result, _fill)
    fills += _fill

    _fill = trapped_ball_fill_multi(result, 2, method=None)
    result = mark_fill(result, _fill)
    fills += _fill

    if use_floodfill:
        _fill = flood_fill_multi(result)
        result = mark_fill(result, _fill)
        fills += _fill

    fillmap = build_fill_map(result, fills)
    fillmap = merge_fill(fillmap)

    return fillmap


def fill_ave(im, fillmap, ave_line=True):
    """
    Average color image based on segmentation region from fillmap.

    :return: average color image (HxWx3)
    """
    im_res = np.copy(im)

    max_fillid = np.max(fillmap)

    for fill_id in range(0, max_fillid + 1):
        mask_index = np.where(fillmap == fill_id)

        # Skip empty mask
        if not len(mask_index[0]) > 0:
            continue

        im_res[mask_index] = np.average(im[mask_index], axis=0)

    if not ave_line:
        im_res[np.where(fillmap == 0)] = (0, 0, 0)

    return im_res


def fill_ave_mask(im, fillmap, threshold=127):
    """
    Average gray image based on segmentation region from fillmap.

    :return: average mask (HxWx1)
    """
    im_res = np.copy(im)

    max_fillid = np.max(fillmap)
    for fill_id in range(1, max_fillid + 1):
        mask_index = np.where(fillmap == fill_id)

        # Skip empty mask
        if not len(mask_index[0]) > 0:
            continue

        color = np.average(im[mask_index], axis=0)
        im_res[mask_index] = 0 if color < threshold else 255

    return im_res


if __name__ == '__main__':
    filename = 'sample'
    im_color = cv2.imread('../input/%s.png' % filename)
    im_lines = cv2.imread('../output/%s.png' % filename, cv2.IMREAD_GRAYSCALE)

    fillmap = fill(im_lines, use_floodfill=True)

    np.save('./result_fill', fillmap)
    cv2.imwrite('./result_fill_raw.png', fillmap)
    cv2.imwrite('./result_fill_vis.png', show_fill_map(fillmap))
    cv2.imwrite('./result_fill_ave.png', fill_ave(im_color, fillmap, ave_line=False))
