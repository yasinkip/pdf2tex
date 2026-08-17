"""
Test modeling layers output sensible results.
"""

from torch import nn

from ..modeling import Encoder, MHALayer, MHAConfig, MLPConfig, NormConfig

_EMB_DIM = 4
_NUM_HEADS = 2
_HID_CHANS = [8]
_NORM_SHAPE = _EMB_DIM

_CORRECT_CONFIGS = {
    "attn_cfg": MHAConfig(_EMB_DIM, _NUM_HEADS),
    "mlp_cfg": MLPConfig(_HID_CHANS),
    "norm_cfg": NormConfig(_NORM_SHAPE),
}

def test_MHALayer_initalises():
    layer = MHALayer(1, 1)

    assert layer is not None
    assert isinstance(layer, nn.Module)


def test_Encoder_initalises():
    encoder = Encoder(2, **_CORRECT_CONFIGS, base_dims=(16, 16, 3), p_dims=(4, 4))

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
        Encoder(2, **_CORRECT_CONFIGS, base_dims=(16, 16, 3), p_dims=(4, 6))
        error = False

    except AssertionError:
        error = True

    assert error


def test_Encoder_with_custom_layers_returns_tensor_iff_tensor_is_correct():
    encoder = Encoder(2, **_CORRECT_CONFIGS, base_dims=(16, 16, 3), p_dims=(4, 4))
    assert encoder is not None
