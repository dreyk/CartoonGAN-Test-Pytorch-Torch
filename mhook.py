import io
import logging

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from network.Transformer import Transformer
import glob
import os

LOG = logging.getLogger(__name__)

PARAMS = {
    'max_size': '1024',
}
models = {}
max_size = 1024
cuda = False

def init_hook(**params):
    LOG.info('Loaded. {}'.format(params))
    global PARAMS
    PARAMS.update(params)
    global max_size
    max_size = PARAMS.get('max_size', '1024')
    max_size = int(max_size)
    LOG.info('Max size {}'.format(max_size))
    if torch.cuda.is_available():
        global cuda
        cuda = True
    LOG.info('Use cuda: {}'.format(cuda))
    for m in glob.glob(PARAMS['model_path'] + '/*.pth'):
        f = os.path.basename(m)
        LOG.info('loading model {} {}'.format(f, m))
        model = Transformer()
        model.load_state_dict(torch.load(m))
        model.eval()
        if cuda:
            model.cuda()
        else:
            model.float()
        global models
        models[f.split('_')[0]] = model

def preprocess(inputs, ctx):
    input_image = Image.open(io.BytesIO(inputs['image'][0])).convert("RGB")
    style = inputs.get('style', ['Shinkai'])[0].decode("utf-8")
    model = models[style]
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h * 1.0 / w
    if h > max_size or w > max_size:
        if ratio > 1:
            h = max_size
            w = int(h * 1.0 / ratio)
        else:
            w = max_size
            h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image, dtype=np.float32)
    input_image = input_image[:, :, [2, 1, 0]]
    # input_image = np.transpose(input_image, (2, 0, 1))
    input_image = -1 + 2 * input_image / 255.0
    # input_image = np.expand_dims(input_image, axis=0)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    if cuda:
        input_image = Variable(input_image, volatile=True).cuda()
    else:
        input_image = Variable(input_image, volatile=True).float()
    output_image = model(input_image)[0]
    output_image = output_image[[2, 1, 0], :, :]
    image = output_image.data.cpu().float().numpy()
    image = (image * 0.5 + 0.5) * 255
    image = np.transpose(image, (1, 2, 0))
    image_bytes = io.BytesIO()
    Image.fromarray(np.uint8(image)).save(image_bytes, format='PNG')
    return {'output': image_bytes.getvalue()}
