from ctypes import POINTER, Structure, c_float, c_int, c_int64, c_size_t, c_void_p

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


llaisysQwen2Model_t = c_void_p


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


def load_qwen2(lib):
    try:
        model_create = lib.llaisysQwen2ModelCreate
        model_destroy = lib.llaisysQwen2ModelDestroy
        model_weights = lib.llaisysQwen2ModelWeights
        model_infer = lib.llaisysQwen2ModelInfer
    except AttributeError:
        return

    model_create.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    model_create.restype = llaisysQwen2Model_t

    model_destroy.argtypes = [llaisysQwen2Model_t]
    model_destroy.restype = None

    model_weights.argtypes = [llaisysQwen2Model_t]
    model_weights.restype = POINTER(LlaisysQwen2Weights)

    model_infer.argtypes = [llaisysQwen2Model_t, POINTER(c_int64), c_size_t]
    model_infer.restype = c_int64
