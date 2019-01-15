import io
import logging

import numpy as np
from PIL import Image


LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')


def preprocess(inputs, ctx):
    input_image = Image.open(io.BytesIO(inputs['image'][0])).convert("RGB")
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = 320
        w = int(h * 1.0 / ratio)
    else:
        w = 320
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = -1 + 2 * input_image
    input_image = np.expand_dims(input_image, axis=0)
    return {'image': input_image}


def postprocess(outputs, ctx):
    image = outputs['0'][0]
    image = (image * 0.5 + 0.5) * 255
    image = np.transpose(image, (1, 2, 0))
    image_bytes = io.BytesIO()
    Image.fromarray(np.uint8(image)).save(image_bytes, format='PNG')
    return {'image': image_bytes.getvalue()}
