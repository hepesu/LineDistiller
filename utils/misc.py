import numpy as np
import cv2
from PIL import Image, ImageDraw
from colorgram import extract
from .fill import fill

KERNEL = np.ones((3, 3), np.uint8)


def shade_color(im, fillmap):
    """
    Compute colorgram based on segmentation region from fillmap.

    :return: colorgrams
    """
    # Denoise image
    s = np.copy(im)
    for _ in range(2):
        s = cv2.bilateralFilter(s, 9, 10, 10)

    color_res = []

    max_fillid = np.max(fillmap)

    for fill_id in range(1, max_fillid + 1):
        mask_index = np.where(fillmap == fill_id)

        # Skip empty mask
        if not len(mask_index[0]) > 0:
            continue

        mask = np.where(fillmap == fill_id, 1, 0).astype(np.uint8)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.boundingRect(contours[0])

        im_mask = mask[:, :, np.newaxis] * im
        im_crop = im_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]

        colors = extract(Image.fromarray(im_crop, 'RGB'), 128)

        # Sort by hue then exclude RGB(0, 0, 0)
        colors.sort(key=lambda c: c.hsl.h)
        c_null = colors[0]
        colors = colors[1:]

        for c in colors:
            c.proportion = c.proportion / (1 - c_null.proportion)

        # Pick by proportion
        colors.sort(key=lambda c: c.proportion, reverse=True)
        colors = colors[:3]

        colors_pick = list(filter(lambda c: c.proportion > 0.2, colors))
        if len(colors_pick) == 0 and len(colors) > 0:
            colors_pick = [colors[0]]

        if len(colors_pick) > 0:
            color_res.append((rect, [tuple(c.rgb) for c in colors_pick]))

    return color_res


def show_color(im, shade_color):
    merged = Image.fromarray(im, 'RGB')
    draw = ImageDraw.Draw(merged)

    for s in shade_color:
        rect, colors = s

        color_width = 10 / len(colors)
        color_height = 5

        color_x = rect[0]
        color_y = rect[1]

        color_size = (int(color_width), color_height)

        for color in colors:
            color = Image.new('RGB', color_size, color)
            merged.paste(color, (int(color_x), color_y))
            color_x += color_width

        draw.rectangle((rect[0], rect[1], rect[0] + 10, rect[1] + 5,), outline="blue")
        draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3],), outline="blue")

    return np.array(merged)


if __name__ == '__main__':
    filename = 'sample'
    im_color = cv2.imread('../input/%s.png' % filename)
    im_lines = cv2.imread('../output/%s.png' % filename, cv2.IMREAD_GRAYSCALE)

    fillmap = fill(im_lines, use_floodfill=False)

    res_shade_color = shade_color(im_color, fillmap)
    lbl_shade_color = show_color(im_color, res_shade_color)

    cv2.imwrite('./result_shade_color.png', lbl_shade_color)
