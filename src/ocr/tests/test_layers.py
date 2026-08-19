"""
Test modeling layers output sensible results.
"""

import torch
import pytest

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


# --- Shape/dimension tests -------------------------------------------------

_MHA_SHAPES = [
    # (batch, num_tokens, embed_dim, num_heads)
    (1, 1, 4, 1),
    (1, 7, 8, 2),
    (3, 5, 8, 8),
    (2, 16, 12, 3),
]


@pytest.mark.parametrize(("B", "N", "D", "H"), _MHA_SHAPES)
def test_MHALayer_preserves_input_shape(B, N, D, H):
    layer = MHALayer(D, H)
    x = torch.randn(B, N, D)

    assert layer(x).shape == (B, N, D)


@pytest.mark.parametrize(("B", "N", "D", "H"), _MHA_SHAPES)
def test_MHALayer_causal_matches_non_causal_shape(B, N, D, H):
    layer = MHALayer(D, H)
    x = torch.randn(B, N, D)

    assert layer(x, causal=True).shape == layer(x, causal=False).shape == (B, N, D)


@pytest.mark.parametrize(("D", "H"), [(4, 1), (8, 2), (12, 3), (8, 8)])
def test_MHALayer_projection_dims(D, H):
    layer = MHALayer(D, H)

    assert layer.head_dim == D // H
    assert layer.head_dim * layer.num_heads == D
    for proj in (layer.Wq, layer.Wk, layer.Wv, layer.out_proj):
        assert proj.weight.shape == (D, D)


def test_MHALayer_kdim_vdim_size_their_projections():
    layer = MHALayer(_EMB_DIM, _NUM_HEADS, kdim=6, vdim=10)

    assert layer.Wk.weight.shape == (_EMB_DIM, 6)
    assert layer.Wv.weight.shape == (_EMB_DIM, 10)
    assert layer.Wq.weight.shape == (_EMB_DIM, _EMB_DIM)


@pytest.mark.parametrize("shape", [(4,), (5, 4), (2, 3, 5, 4)])
def test_MHALayer_rejects_non_3D_input(shape):
    layer = MHALayer(_EMB_DIM, _NUM_HEADS)

    with pytest.raises(AssertionError):
        layer(torch.randn(*shape))


_ENCODER_DIMS = [
    # (base_dims, p_dims, expected_num_patches)
    ((16, 16, 3), (4, 4), 16),
    ((16, 16, 1), (8, 8), 4),
    ((32, 32, 3), (4, 4), 64),
    ((12, 12, 5), (6, 6), 4),
]


@pytest.mark.parametrize(("base_dims", "p_dims", "N"), _ENCODER_DIMS)
def test_Encoder_flattened_dims_match_patch_grid(base_dims, p_dims, N):
    encoder = Encoder(1, **_CORRECT_CONFIGS, base_dims=base_dims, p_dims=p_dims)
    pH, pW = p_dims
    C = base_dims[2]

    assert encoder.flattened_dims == (N, pH * pW * C)


@pytest.mark.parametrize(("base_dims", "p_dims", "N"), _ENCODER_DIMS)
def test_Encoder_token_parameter_dims(base_dims, p_dims, N):
    encoder = Encoder(1, **_CORRECT_CONFIGS, base_dims=base_dims, p_dims=p_dims)

    # one [CLS] token is prepended, so positions cover N + 1 tokens
    assert encoder.cls.shape == (_EMB_DIM,)
    assert encoder.positional.shape == (N + 1, _EMB_DIM)


@pytest.mark.parametrize(("base_dims", "p_dims", "N"), _ENCODER_DIMS)
def test_Encoder_embedder_maps_flattened_patch_to_embed_dim(base_dims, p_dims, N):
    encoder = Encoder(1, **_CORRECT_CONFIGS, base_dims=base_dims, p_dims=p_dims)
    patches = torch.randn(2, *encoder.flattened_dims)

    assert encoder.embedder.weight.shape == (_EMB_DIM, encoder.flattened_dims[1])
    assert encoder.embedder(patches).shape == (2, N, _EMB_DIM)


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_Encoder_stack_preserves_token_shape(num_blocks):
    # the MLP must project back to embed_dim for blocks to be stackable
    configs = {**_CORRECT_CONFIGS, "mlp_cfg": MLPConfig([8, _EMB_DIM], activation_layer=nn.GELU)}
    encoder = Encoder(num_blocks, **configs, base_dims=(16, 16, 3), p_dims=(4, 4))
    N = encoder.flattened_dims[0]
    tokens = torch.randn(2, N + 1, _EMB_DIM)

    assert encoder.stack(tokens).shape == (2, N + 1, _EMB_DIM)


@pytest.mark.parametrize("shape", [(16, 16, 3), (2, 16, 16, 3, 1), (16, 3)])
def test_Encoder_forward_rejects_non_4D_input(shape):
    encoder = Encoder(1, **_CORRECT_CONFIGS, base_dims=(16, 16, 3), p_dims=(4, 4))

    with pytest.raises(AssertionError):
        encoder(torch.randn(*shape))


@pytest.mark.parametrize("shape", [(2, 32, 32, 3), (2, 16, 16, 1), (2, 16, 32, 3)])
def test_Encoder_forward_rejects_mismatched_image_dims(shape):
    encoder = Encoder(1, **_CORRECT_CONFIGS, base_dims=(16, 16, 3), p_dims=(4, 4))

    with pytest.raises(AssertionError):
        encoder(torch.randn(*shape))


@pytest.mark.parametrize(("base_dims", "p_dims", "N"), _ENCODER_DIMS)
def test_Encoder_forward_returns_one_token_per_patch_plus_cls(base_dims, p_dims, N):
    configs = {**_CORRECT_CONFIGS, "mlp_cfg": MLPConfig([8, _EMB_DIM], activation_layer=nn.GELU)}
    encoder = Encoder(2, **configs, base_dims=base_dims, p_dims=p_dims)
    images = torch.randn(2, *base_dims)

    assert encoder(images).shape == (2, N + 1, _EMB_DIM)
