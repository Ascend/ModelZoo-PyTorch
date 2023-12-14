import os
import torch.nn as nn
from .efficientnet_builder import decode_arch_def, resolve_bn_args
from .mobilenetv3 import MobileNetV3, MobileNetV3Features, build_model_with_cfg, default_cfg_for_features
from .layers import hard_sigmoid
from .efficientnet_blocks import resolve_act_layer
from .registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(CURRENT_PATH, '../../url.ini'), 'r') as _f:
    _content = _f.read()
    hardcorenas_a_url = _content.split('hardcorenas_a_url=')[1].split('\n')[0]
    hardcorenas_b_url = _content.split('hardcorenas_b_url=')[1].split('\n')[0]
    hardcorenas_c_url = _content.split('hardcorenas_c_url=')[1].split('\n')[0]
    hardcorenas_d_url = _content.split('hardcorenas_d_url=')[1].split('\n')[0]
    hardcorenas_e_url = _content.split('hardcorenas_e_url=')[1].split('\n')[0]
    hardcorenas_f_url = _content.split('hardcorenas_f_url=')[1].split('\n')[0]


default_cfgs = {
    'hardcorenas_a': _cfg(url=hardcorenas_a_url),
    'hardcorenas_b': _cfg(url=hardcorenas_b_url),
    'hardcorenas_c': _cfg(url=hardcorenas_c_url),
    'hardcorenas_d': _cfg(url=hardcorenas_d_url),
    'hardcorenas_e': _cfg(url=hardcorenas_e_url),
    'hardcorenas_f': _cfg(url=hardcorenas_f_url),
}


def _gen_hardcorenas(pretrained, variant, arch_def, **kwargs):
    """Creates a hardcorenas model

    Ref impl: https://github.com/Alibaba-MIIL/HardCoReNAS
    Paper: https://arxiv.org/abs/2102.11646

    """
    num_features = 1280

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=32,
        channel_multiplier=1,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )

    features_only = False
    model_cls = MobileNetV3
    kwargs_filter = None
    if model_kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'global_pool', 'head_conv', 'head_bias', 'global_pool')
        model_cls = MobileNetV3Features
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **model_kwargs)
    if features_only:
        model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model


@register_model
def hardcorenas_a(pretrained=False, **kwargs):
    """ hardcorenas_A """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'],
                ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_a', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_b(pretrained=False, **kwargs):
    """ hardcorenas_B """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'],
                ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25', 'ir_r1_k3_s1_e3_c24_nre'],
                ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre'],
                ['ir_r1_k5_s2_e3_c80', 'ir_r1_k5_s1_e3_c80', 'ir_r1_k3_s1_e3_c80', 'ir_r1_k3_s1_e3_c80'],
                ['ir_r1_k5_s1_e3_c112', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k3_s1_e3_c192_se0.25'],
                ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_b', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_c(pretrained=False, **kwargs):
    """ hardcorenas_C """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre', 'ir_r1_k5_s1_e3_c40_nre',
                 'ir_r1_k5_s1_e3_c40_nre'],
                ['ir_r1_k5_s2_e4_c80', 'ir_r1_k5_s1_e6_c80_se0.25', 'ir_r1_k3_s1_e3_c80', 'ir_r1_k3_s1_e3_c80'],
                ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112', 'ir_r1_k3_s1_e3_c112'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k3_s1_e3_c192_se0.25'],
                ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_c', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_d(pretrained=False, **kwargs):
    """ hardcorenas_D """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre_se0.25', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e3_c40_nre_se0.25', 'ir_r1_k5_s1_e4_c40_nre_se0.25', 'ir_r1_k3_s1_e3_c40_nre_se0.25'],
                ['ir_r1_k5_s2_e4_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25',
                 'ir_r1_k3_s1_e3_c80_se0.25'],
                ['ir_r1_k3_s1_e4_c112_se0.25', 'ir_r1_k5_s1_e4_c112_se0.25', 'ir_r1_k3_s1_e3_c112_se0.25',
                 'ir_r1_k5_s1_e3_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
                 'ir_r1_k3_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_d', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_e(pretrained=False, **kwargs):
    """ hardcorenas_E """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre_se0.25', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c40_nre_se0.25', 'ir_r1_k5_s1_e4_c40_nre_se0.25', 'ir_r1_k5_s1_e4_c40_nre_se0.25',
                 'ir_r1_k3_s1_e3_c40_nre_se0.25'], ['ir_r1_k5_s2_e4_c80_se0.25', 'ir_r1_k3_s1_e6_c80_se0.25'],
                ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25',
                 'ir_r1_k5_s1_e3_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
                 'ir_r1_k3_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_e', arch_def=arch_def, **kwargs)
    return model


@register_model
def hardcorenas_f(pretrained=False, **kwargs):
    """ hardcorenas_F """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre_se0.25', 'ir_r1_k5_s1_e3_c24_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c40_nre_se0.25', 'ir_r1_k5_s1_e6_c40_nre_se0.25'],
                ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25', 'ir_r1_k3_s1_e3_c80_se0.25',
                 'ir_r1_k3_s1_e3_c80_se0.25'],
                ['ir_r1_k3_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25',
                 'ir_r1_k3_s1_e3_c112_se0.25'],
                ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k3_s1_e6_c192_se0.25',
                 'ir_r1_k3_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]
    model = _gen_hardcorenas(pretrained=pretrained, variant='hardcorenas_f', arch_def=arch_def, **kwargs)
    return model
