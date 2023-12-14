"""Pytorch impl of MxNet Gluon ResNet/(SE)ResNeXt variants
This file evolved from https://github.com/pytorch/vision 'resnet.py' with (SE)-ResNeXt additions
and ports of Gluon variations (https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/resnet.py) 
by Ross Wightman
"""
import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import SEModule
from .registry import register_model
from .resnet import ResNet, Bottleneck, BasicBlock


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(CURRENT_PATH, '../../url.ini'), 'r') as _f:
    _content = _f.read()
    gluon_resnet18_v1b_0757602b_url = _content.split('gluon_resnet18_v1b_0757602b_url=')[1].split('\n')[0]
    gluon_resnet34_v1b_c6d82d59_url = _content.split('gluon_resnet34_v1b_c6d82d59_url=')[1].split('\n')[0]
    gluon_resnet50_v1b_0ebe02e2_url = _content.split('gluon_resnet50_v1b_0ebe02e2_url=')[1].split('\n')[0]
    gluon_resnet101_v1b_3b017079_url = _content.split('gluon_resnet101_v1b_3b017079_url=')[1].split('\n')[0]
    gluon_resnet152_v1b_c1edb0dd_url = _content.split('gluon_resnet152_v1b_c1edb0dd_url=')[1].split('\n')[0]
    gluon_resnet50_v1c_48092f55_url = _content.split('gluon_resnet50_v1c_48092f55_url=')[1].split('\n')[0]
    gluon_resnet101_v1c_1f26822a_url = _content.split('gluon_resnet101_v1c_1f26822a_url=')[1].split('\n')[0]
    gluon_resnet152_v1c_a3bb0b98_url = _content.split('gluon_resnet152_v1c_a3bb0b98_url=')[1].split('\n')[0]
    gluon_resnet50_v1d_818a1b1b_url = _content.split('gluon_resnet50_v1d_818a1b1b_url=')[1].split('\n')[0]
    gluon_resnet101_v1d_0f9c8644_url = _content.split('gluon_resnet101_v1d_0f9c8644_url=')[1].split('\n')[0]
    gluon_resnet152_v1d_bd354e12_url = _content.split('gluon_resnet152_v1d_bd354e12_url=')[1].split('\n')[0]
    gluon_resnet50_v1s_1762acc0_url = _content.split('gluon_resnet50_v1s_1762acc0_url=')[1].split('\n')[0]
    gluon_resnet101_v1s_60fe0cc1_url = _content.split('gluon_resnet101_v1s_60fe0cc1_url=')[1].split('\n')[0]
    gluon_resnet152_v1s_dcc41b81_url = _content.split('gluon_resnet152_v1s_dcc41b81_url=')[1].split('\n')[0]
    gluon_resnext50_32x4d_e6a097c1_url = _content.split('gluon_resnext50_32x4d_e6a097c1_url=')[1].split('\n')[0]
    gluon_resnext101_32x4d_b253c8c4_url = _content.split('gluon_resnext101_32x4d_b253c8c4_url=')[1].split('\n')[0]
    gluon_resnext101_64x4d_f9a8e184_url = _content.split('gluon_resnext101_64x4d_f9a8e184_url=')[1].split('\n')[0]
    gluon_seresnext50_32x4d_90cf2d6e_url = _content.split('gluon_seresnext50_32x4d_90cf2d6e_url=')[1].split('\n')[0]
    gluon_seresnext101_32x4d_cf52900d_url = _content.split('gluon_seresnext101_32x4d_cf52900d_url=')[1].split('\n')[0]
    gluon_seresnext101_64x4d_f9926f93_url = _content.split('gluon_seresnext101_64x4d_f9926f93_url=')[1].split('\n')[0]
    gluon_senet154_70a1a3c0_url = _content.split('gluon_senet154_70a1a3c0_url=')[1].split('\n')[0]


default_cfgs = {
    'gluon_resnet18_v1b': _cfg(url=gluon_resnet18_v1b_0757602b_url),
    'gluon_resnet34_v1b': _cfg(url=gluon_resnet34_v1b_c6d82d59_url),
    'gluon_resnet50_v1b': _cfg(url=gluon_resnet50_v1b_0ebe02e2_url),
    'gluon_resnet101_v1b': _cfg(url=gluon_resnet101_v1b_3b017079_url),
    'gluon_resnet152_v1b': _cfg(url=gluon_resnet152_v1b_c1edb0dd_url),
    'gluon_resnet50_v1c': _cfg(url=gluon_resnet50_v1c_48092f55_url,
                               first_conv='conv1.0'),
    'gluon_resnet101_v1c': _cfg(url=gluon_resnet101_v1c_1f26822a_url,
                                first_conv='conv1.0'),
    'gluon_resnet152_v1c': _cfg(url=gluon_resnet152_v1c_a3bb0b98_url,
                                first_conv='conv1.0'),
    'gluon_resnet50_v1d': _cfg(url=gluon_resnet50_v1d_818a1b1b_url,
                               first_conv='conv1.0'),
    'gluon_resnet101_v1d': _cfg(url=gluon_resnet101_v1d_0f9c8644_url,
                                first_conv='conv1.0'),
    'gluon_resnet152_v1d': _cfg(url=gluon_resnet152_v1d_bd354e12_url,
                                first_conv='conv1.0'),
    'gluon_resnet50_v1s': _cfg(url=gluon_resnet50_v1s_1762acc0_url,
                               first_conv='conv1.0'),
    'gluon_resnet101_v1s': _cfg(url=gluon_resnet101_v1s_60fe0cc1_url,
                                first_conv='conv1.0'),
    'gluon_resnet152_v1s': _cfg(url=gluon_resnet152_v1s_dcc41b81_url,
                                first_conv='conv1.0'),
    'gluon_resnext50_32x4d': _cfg(url=gluon_resnext50_32x4d_e6a097c1_url),
    'gluon_resnext101_32x4d': _cfg(url=gluon_resnext101_32x4d_b253c8c4_url),
    'gluon_resnext101_64x4d': _cfg(url=gluon_resnext101_64x4d_f9a8e184_url),
    'gluon_seresnext50_32x4d': _cfg(url=gluon_seresnext50_32x4d_90cf2d6e_url),
    'gluon_seresnext101_32x4d': _cfg(url=gluon_seresnext101_32x4d_cf52900d_url),
    'gluon_seresnext101_64x4d': _cfg(url=gluon_seresnext101_64x4d_f9926f93_url),
    'gluon_senet154': _cfg(url=gluon_senet154_70a1a3c0_url,
                           first_conv='conv1.0'),
}


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)


@register_model
def gluon_resnet18_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('gluon_resnet18_v1b', pretrained, **model_args)


@register_model
def gluon_resnet34_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('gluon_resnet34_v1b', pretrained, **model_args)


@register_model
def gluon_resnet50_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('gluon_resnet50_v1b', pretrained, **model_args)


@register_model
def gluon_resnet101_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('gluon_resnet101_v1b', pretrained, **model_args)


@register_model
def gluon_resnet152_v1b(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('gluon_resnet152_v1b', pretrained, **model_args)


@register_model
def gluon_resnet50_v1c(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet50_v1c', pretrained, **model_args)


@register_model
def gluon_resnet101_v1c(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet101_v1c', pretrained, **model_args)


@register_model
def gluon_resnet152_v1c(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet152_v1c', pretrained, **model_args)


@register_model
def gluon_resnet50_v1d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('gluon_resnet50_v1d', pretrained, **model_args)


@register_model
def gluon_resnet101_v1d(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('gluon_resnet101_v1d', pretrained, **model_args)


@register_model
def gluon_resnet152_v1d(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    return _create_resnet('gluon_resnet152_v1d', pretrained, **model_args)


@register_model
def gluon_resnet50_v1s(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet50_v1s', pretrained, **model_args)



@register_model
def gluon_resnet101_v1s(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet101_v1s', pretrained, **model_args)


@register_model
def gluon_resnet152_v1s(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type='deep', **kwargs)
    return _create_resnet('gluon_resnet152_v1s', pretrained, **model_args)



@register_model
def gluon_resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('gluon_resnext50_32x4d', pretrained, **model_args)


@register_model
def gluon_resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('gluon_resnext101_32x4d', pretrained, **model_args)


@register_model
def gluon_resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4, **kwargs)
    return _create_resnet('gluon_resnext101_64x4d', pretrained, **model_args)


@register_model
def gluon_seresnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a SEResNeXt50-32x4d model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_seresnext50_32x4d', pretrained, **model_args)


@register_model
def gluon_seresnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a SEResNeXt-101-32x4d model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_seresnext101_32x4d', pretrained, **model_args)


@register_model
def gluon_seresnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a SEResNeXt-101-64x4d model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4,
        block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_seresnext101_64x4d', pretrained, **model_args)


@register_model
def gluon_senet154(pretrained=False, **kwargs):
    """Constructs an SENet-154 model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4, stem_type='deep',
        down_kernel_size=3, block_reduce_first=2, block_args=dict(attn_layer=SEModule), **kwargs)
    return _create_resnet('gluon_senet154', pretrained, **model_args)
