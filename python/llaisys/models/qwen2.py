from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DataType, DeviceType
from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights, llaisysDeviceType_t, llaisysQwen2Model_t

import json
from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        meta = self._load_meta(model_path)
        device_ids = (c_int * 1)(0)

        self._model: llaisysQwen2Model_t = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            llaisysDeviceType_t(device),
            device_ids,
            c_int(1),
        )
        self._meta = meta

        self._weight: LlaisysQwen2Weights = LIB_LLAISYS.llaisysQwen2ModelWeights(
            self._model
        ).contents

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                dst = self._weight_tensor_for_name(name_)
                if not dst:
                    raise RuntimeError(f"Weight tensor is not initialized: {name_}")
                LIB_LLAISYS.tensorLoad(dst, c_void_p(tensor.data_ptr()))

    @staticmethod
    def _load_meta(model_path: Path) -> LlaisysQwen2Meta:
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        dtype_map = {
            "bfloat16": DataType.BF16,
            "float16": DataType.F16,
            "float32": DataType.F32,
        }

        meta = LlaisysQwen2Meta()
        meta.dtype = dtype_map[config["torch_dtype"]]
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = meta.hs // meta.nh
        meta.di = config["intermediate_size"]
        meta.maxseq = config["max_position_embeddings"]
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config["rope_theta"]
        meta.end_token = config["eos_token_id"]
        return meta

    def _weight_tensor_for_name(self, name: str):
        if name == "model.embed_tokens.weight":
            return self._weight.in_embed
        if name == "lm_head.weight":
            return self._weight.out_embed
        if name == "model.norm.weight":
            return self._weight.out_norm_w

        parts = name.split(".")
        if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
            raise KeyError(f"Unknown Qwen2 weight: {name}")

        layer = int(parts[2])
        suffix = ".".join(parts[3:])

        if suffix == "input_layernorm.weight":
            return self._weight.attn_norm_w[layer]
        if suffix == "self_attn.q_proj.weight":
            return self._weight.attn_q_w[layer]
        if suffix == "self_attn.q_proj.bias":
            return self._weight.attn_q_b[layer]
        if suffix == "self_attn.k_proj.weight":
            return self._weight.attn_k_w[layer]
        if suffix == "self_attn.k_proj.bias":
            return self._weight.attn_k_b[layer]
        if suffix == "self_attn.v_proj.weight":
            return self._weight.attn_v_w[layer]
        if suffix == "self_attn.v_proj.bias":
            return self._weight.attn_v_b[layer]
        if suffix == "self_attn.o_proj.weight":
            return self._weight.attn_o_w[layer]
        if suffix == "post_attention_layernorm.weight":
            return self._weight.mlp_norm_w[layer]
        if suffix == "mlp.gate_proj.weight":
            return self._weight.mlp_gate_w[layer]
        if suffix == "mlp.up_proj.weight":
            return self._weight.mlp_up_w[layer]
        if suffix == "mlp.down_proj.weight":
            return self._weight.mlp_down_w[layer]

        raise KeyError(f"Unknown Qwen2 weight: {name}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        output_tokens = list(inputs)
        steps = self._meta.maxseq if max_new_tokens is None else max_new_tokens

        for _ in range(steps):
            token_ids = (c_int64 * len(output_tokens))(*output_tokens)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_ids,
                c_size_t(len(output_tokens)),
            )
            output_tokens.append(int(next_token))

            if next_token == self._meta.end_token:
                break

        return output_tokens
