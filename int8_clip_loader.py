import torch
import logging

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

import nodes  # for reusing the stock CLIPLoader "type" dropdown

from .int8_quant import Int8TensorwiseOps, INT8ModelPatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ComfyUI historically used the "clip" folder key and later added
# "text_encoders". We union both so files show up regardless of layout.
_TE_FOLDER_KEYS = ("text_encoders", "clip")


def _te_filename_list():
    files = []
    for key in _TE_FOLDER_KEYS:
        try:
            files += folder_paths.get_filename_list(key)
        except Exception:
            pass
    return sorted(set(files))


def _te_full_path(name):
    for key in _TE_FOLDER_KEYS:
        try:
            p = folder_paths.get_full_path(key, name)
            if p is not None:
                return p
        except Exception:
            pass
    # Last resort: let comfy raise a helpful error under the canonical key.
    return folder_paths.get_full_path("text_encoders", name)


_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _prime_int8_ops(weight_dtype, on_the_fly_quantization, enable_convrot,
                    lora_mode, excluded_names):
    """
    Reset the shared Int8TensorwiseOps class-level flags before a load.

    These flags are GLOBAL (class attributes) and are also touched by the
    diffusion loader, so we must set every one of them here or a previous
    diffusion load could leak stale state (LoRA patches, exclusions, etc.)
    into the text-encoder load.
    """
    lora_mode = str(lora_mode)
    if lora_mode not in {"None", "Stochastic", "Dynamic"}:
        lora_mode = "None"

    Int8TensorwiseOps.excluded_names = list(excluded_names or [])
    Int8TensorwiseOps.dynamic_quantize = bool(on_the_fly_quantization)
    Int8TensorwiseOps.enable_convrot = bool(enable_convrot)
    Int8TensorwiseOps.use_triton = True
    Int8TensorwiseOps._is_prequantized = False
    Int8TensorwiseOps.lora_mode = lora_mode
    Int8TensorwiseOps.dynamic_lora = lora_mode == "Dynamic"
    Int8TensorwiseOps.dynamic_load_device = None
    Int8TensorwiseOps.lora_patches = {}          # critical: don't inherit UNet LoRA patches
    Int8TensorwiseOps.compute_dtype = _DTYPE_MAP.get(str(weight_dtype), None)

    if hasattr(Int8TensorwiseOps, "_logged_otf"):
        delattr(Int8TensorwiseOps, "_logged_otf")


def _load_int8_clip(clip_paths, clip_type, weight_dtype, on_the_fly_quantization,
                    enable_convrot, lora_mode, excluded_names):
    # For a pre-quantized file each layer already carries its own
    # weight/weight_scale (+ optional comfy_quant convrot metadata), so
    # exclusions are only used by the on-the-fly path.
    _prime_int8_ops(weight_dtype, on_the_fly_quantization, enable_convrot,
                    lora_mode, excluded_names)

    state_dicts = []
    for p in clip_paths:
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        if "scaled_fp8" in sd:
            raise NotImplementedError(
                "This text encoder is scaled-FP8. INT8 custom ops can't be mixed "
                "with scaled-FP8 in the same encoder; use the stock CLIP loader "
                f"for this file:\n{p}"
            )
        state_dicts.append(sd)

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=clip_type,
        state_dicts=state_dicts,
        model_options={
            "custom_operations": Int8TensorwiseOps,
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        },
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    # Wrap the text-encoder patcher so INT8 LoRA bake / dynamic paths work,
    # exactly like the diffusion node does. With no matching patches this
    # falls straight through to the stock ModelPatcher behaviour.
    clip.patcher = INT8ModelPatcher.clone(clip.patcher)
    return clip


def _clip_type_from_str(type_str):
    return getattr(comfy.sd.CLIPType, str(type_str).upper(),
                   comfy.sd.CLIPType.STABLE_DIFFUSION)


# Minimal default exclusions for ON-THE-FLY text-encoder quantization.
# Embeddings / norms are already skipped by the ops layer; this just keeps a
# few numerically sensitive projections in high precision. Tune per model.
# (Ignored entirely when loading an already-quantized file.)
_TE_DEFAULT_EXCLUSIONS = [
    "shared", "embed_tokens", "token_embedding",
    "relative_attention_bias",
    "final_layer_norm", "encoder.final_layer_norm",
    "logit_scale", "text_projection",
]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class CLIPLoaderINT8:
    """
    Load a single INT8 text encoder (e.g. umt5_xxl_int8_convrot.safetensors)
    using the same Int8TensorwiseOps custom operations as the diffusion loader,
    but through ComfyUI's text-encoder loading path.
    """

    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (_te_filename_list(),),
                # reuse the stock dropdown so every clip_type (wan, flux, ...) is available
                "type": base["required"]["type"],
                "weight_dtype": (["default", "fp16", "bf16", "fp32"],
                                 {"tooltip": "INT8 compute dtype. 'default' follows the encoder dtype."}),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False, "tooltip": "Quantize a bf16/fp16 encoder to INT8 at load. Leave OFF for an already-INT8 file."}),
                "enable_convrot": ("BOOLEAN", {"default": True, "tooltip": "ConvRot rotation for on-the-fly quant. Pre-quantized files carry their own convrot flag per-layer and ignore this."}),
                "lora_mode": (["None", "Stochastic", "Dynamic"], {"default": "None"}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    TITLE = "Load CLIP INT8 (W8A8)"
    DESCRIPTION = "Load an INT8 text encoder with fast triton inference."

    def load_clip(self, clip_name, type="wan", weight_dtype="default",
                  on_the_fly_quantization=False, enable_convrot=True, lora_mode="None"):
        clip_path = _te_full_path(clip_name)
        clip_type = _clip_type_from_str(type)
        excl = _TE_DEFAULT_EXCLUSIONS if on_the_fly_quantization else []
        clip = _load_int8_clip(
            [clip_path], clip_type, weight_dtype, on_the_fly_quantization,
            enable_convrot, lora_mode, excl,
        )
        return (clip,)


class DualCLIPLoaderINT8(CLIPLoaderINT8):
    """
    Two-encoder variant (e.g. Flux: clip_l + t5xxl, or HiDream, SD3, etc.).
    Either slot may be a plain bf16/fp16 or an INT8 file; each layer is
    detected independently by its own weight/weight_scale.
    """

    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        names = _te_filename_list()
        return {
            "required": {
                "clip_name1": (names,),
                "clip_name2": (names,),
                "type": base["required"]["type"],
                "weight_dtype": (["default", "fp16", "bf16", "fp32"],),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False}),
                "enable_convrot": ("BOOLEAN", {"default": True}),
                "lora_mode": (["None", "Stochastic", "Dynamic"], {"default": "None"}),
            },
        }

    TITLE = "Load Dual CLIP INT8 (W8A8)"

    def load_clip(self, clip_name1, clip_name2, type="flux", weight_dtype="default",
                  on_the_fly_quantization=False, enable_convrot=True, lora_mode="None"):
        paths = [_te_full_path(clip_name1), _te_full_path(clip_name2)]
        clip_type = _clip_type_from_str(type)
        excl = _TE_DEFAULT_EXCLUSIONS if on_the_fly_quantization else []
        clip = _load_int8_clip(
            paths, clip_type, weight_dtype, on_the_fly_quantization,
            enable_convrot, lora_mode, excl,
        )
        return (clip,)
