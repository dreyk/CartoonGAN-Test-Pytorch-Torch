import io
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import glob
import os

LOG = logging.getLogger(__name__)

PARAMS = {
    'max_size': '1024',
    'model_path': '/model/21styles.model',
    'styles_path': '/model/21styles'
}
style_model = None
max_size = 1024
cuda = False
styles = {}

shornames = {'mosaic': 'mosaic', 'picasso': 'picasso_selfport1907', 'robert': 'robert', 'candy': 'candy', 'scream': 'the_scream', 'composition': 'composition_vii', 'shipwreck': 'shipwreck', 'rain': 'rain_princess', 'mosaic1': 'mosaic_ducks_massimo', 'escher': 'escher_sphere', 'udnie': 'udnie', 'wave': 'wave', 'woman': 'woman-with-hat-matisse', 'la_muse': 'la_muse', 'seated': 'seated-nude', 'pencil': 'pencil', 'strip': 'strip', 'feathers': 'feathers', 'starry': 'starry_night', 'stars2': 'stars2', 'frida': 'frida_kahlo'}


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
    global style_model
    style_model = Net(ngf=128)
    LOG.info('Loading model: {}'.format(PARAMS['model_path']))
    state = torch.load(PARAMS['model_path'])
    new_state = {}
    for k,v in state.items():
        ks = k.split('.')
        if ks[-1]=='running_mean' or ks[-1]=='running_var':
            continue
        new_state[k]=v
    style_model.load_state_dict(new_state)
    if cuda:
        style_model.cuda()
    else:
        style_model.float()
    for m in glob.glob(PARAMS['styles_path'] + '/*.jpg'):
        img = Image.open(m)
        style = tensor_load_rgbimage(img, size=max_size).unsqueeze(0)
        style = preprocess_batch(style)
        if cuda:
            style = style.cuda()
        global styles
        f = os.path.basename(m)
        f = f.split('.')[0]
        LOG.info('Add style: {}'.format(f))
        styles[f] = style


def preprocess(inputs, ctx):
    content_image = Image.open(io.BytesIO(inputs['image'][0]))
    w = content_image.size[0]
    h = content_image.size[1]
    if w > h:
        if w > 1024:
            ratio = float(w) / 1024.0
            w = 1024
            h = float(h/ratio)
    else:
        if h > 1024:
            ratio = float(h) / 1024.0
            h = 1024
            w = float(w/ratio)
    h = int(h)
    w = int(w)

    content_image = tensor_load_rgbimage(content_image, size=max_size, keep_asp=True).unsqueeze(0)
    content_image = preprocess_batch(content_image)
    if cuda:
        content_image = content_image.cuda()
    style = inputs.get('style', ['candy'])[0].decode("utf-8")
    style = shornames.get(style,style)
    style = styles[style]
    style_v = Variable(style)
    content_image = Variable(content_image)
    style_model.setTarget(style_v)
    output = style_model(content_image)
    image = tensor_save_bgrimage(output.data[0], cuda)
    image_bytes = io.BytesIO()
    image = image.resize((w, h), Image.BICUBIC)
    image.save(image_bytes, format='PNG')
    return {'output': image_bytes.getvalue()}


# define Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


# proposed Inspiration(CoMatch) Layer
class Inspiration(nn.Module):
    """ Inspiration Layer (from MSG-Net paper)
    tuning the featuremap with target Gram Matrix
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # non-parameter buffer
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x ' + str(self.C) + ')'


# some basic layers, with reflectance padding
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Bottleneck(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                            kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


# the MSG-Net
class Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=[]):
        super(Net, self).__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = []
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                   norm_layer(64),
                   nn.ReLU(inplace=True),
                   block(64, 32, 2, 1, norm_layer),
                   block(32 * expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)

        model = []
        self.ins = Inspiration(ngf * expansion)
        model += [self.model1]
        model += [self.ins]

        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

        model += [upblock(ngf * expansion, 32, 2, norm_layer),
                  upblock(32 * expansion, 16, 2, norm_layer),
                  norm_layer(16 * expansion),
                  nn.ReLU(inplace=True),
                  ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def setTarget(self, Xs):
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def forward(self, input):
        return self.model(input)


def tensor_load_rgbimage(img, size=None, scale=None, keep_asp=False):
    img = img.convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    return img


def tensor_save_bgrimage(tensor, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    return tensor_save_rgbimage(tensor, cuda)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


#pencil,la_muse,candy,composition_vii,escher_sphere,feathers,frida_kahlo,mosaic,mosaic_ducks_massimo,picasso_selfport1907,rain_princess,robert,seated-nude,shipwreck,starry_night,stars2,strip,the_scream,udnie,wave,woman-with-hat-matisse