"""
Test modeling layers output sensible results.
"""

from torch import nn

from ..modeling import Encoder, MHALayer, MHAConfig, MLPConfig, NormConfig


def test_MHALayer_initalises():
    layer = MHALayer(1, 1)

    assert layer is not None
    assert isinstance(layer, nn.MultiheadAttention)


def test_Encoder_initalises():
    encoder = Encoder()

    assert encoder is not None
    assert isinstance(encoder, nn.Module)


def test_MHALayer_fails_if_incompatible_dims():
    try:
        MHALayer(16, 3)
        error = False

    except AssertionError:
        error = True

    assert error


def test_Encoder_fails_if_incompatible_image_shape():
    try:
        Encoder(1, 16, base_dims=(16, 16, 3), p_dims=(4, 6, 3))
        error = True

    except AssertionError:
        error = False

    assert error


def test_Encoder_fails_if_incompatible_channel_dim():
    try:
        Encoder(1, 16, base_dims=(16, 16, 3), p_dims=(4, 4, 2))
        error = True

    except AssertionError:
        error = False

    assert error


def test_Encoder_with_custom_layers_returns_tensor_iff_tensor_is_correct():
    D = 16
    norm_cfg = NormConfig()
    attn_cfg = MHAConfig(D, 4)
    mlp_cfg = MLPConfig([8])

    encoder = Encoder(1, attn_cfg, mlp_cfg, (16, 16, 3), (4, 4, 3), norm_cfg=norm_cfg)

    assert encoder is not None
