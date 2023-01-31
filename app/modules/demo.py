import cv2
import math
import numpy as np
import glob
import os
import io
from PIL import Image


"""顯示結果"""
def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

from PIL import Image, ImageDraw, ImageFont
def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="demo/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
#         draw_left.polygon(box, fill=color)
        draw_left.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

def points_offset(points, offset_w, offset_h):
    offset_points = []
    for p in points:
        offset_points.append([p[0]+offset_w, p[1]+offset_h])
    return offset_points

def sharpen(path):    
    sigma = 100

    stream = io.BytesIO()
    pil_img = Image.open(path)
    
    if pil_img.mode == '1':
        pil_img.convert('RGB')

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    pil_img = buf.getvalue()

    stream.close()

    decode = cv2.imdecode(np.frombuffer(pil_img, np.uint8), cv2.IMREAD_COLOR)

    # img = np.frombuffer(image, dtype=np.uint8)
    # img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    blur_img = cv2.GaussianBlur(decode, (0, 0), sigma)
    res_img = cv2.addWeighted(decode, 1.5, blur_img, -0.6, 0)

    _, encoded_image = cv2.imencode(f'.tif', res_img)
    img_bytes = encoded_image.tobytes()
    return img_bytes
