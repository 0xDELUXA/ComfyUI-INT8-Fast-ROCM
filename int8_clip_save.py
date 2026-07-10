import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import json
import os
import logging
import torch
from comfy.cli_args import args


def _is_dynamic_lora_enabled():
    try:
        from .int8_quant import Int8TensorwiseOps
        return bool(getattr(Int8TensorwiseOps, "dynamic_lora", False))
    except Exception:
        return False


class INT8CLIPSave:
    """
    Save a CLIP/text-encoder patcher that was loaded (or on-the-fly quantized)
    via CLIPLoaderINT8 / DualCLIPLoaderINT8 back out as proper INT8 file(s).

    This mirrors INT8ModelSave's pre-pass (force-load, LoRA bake, comfy_quant /
    weight_scale extra_keys, LazyCastingParam bypass) but targets clip.patcher
    instead of the diffusion model patcher, and reuses stock CLIPSave's
    per-encoder split/prefix-strip so output files match the layout the
    community's pre-quantized CLIP files already use (and what
    CLIPLoaderINT8/DualCLIPLoaderINT8 expect to read back).
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",),
                              "filename_prefix": ("STRING", {"default": "int8_clip/INT8_CLIP"}), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "loaders"

    def save(self, clip, filename_prefix, prompt=None, extra_pnginfo=None):
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        model_patcher = clip.patcher

        extra_keys = {}
        patched_modules = []
        patched_module_ids = set()

        def mark_module_for_direct_save(module):
            module_id = id(module)
            if module_id in patched_module_ids:
                return
            had_flag = hasattr(module, "comfy_patched_weights")
            old_flag = getattr(module, "comfy_patched_weights", False)
            patched_modules.append((module, had_flag, old_flag))
            patched_module_ids.add(module_id)
            module.comfy_patched_weights = True

        def module_has_int8_param(module):
            for attr in ("weight", "bias"):
                tensor = getattr(module, attr, None)
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int8:
                    return True
            return False

        def iter_model_modules(patcher):
            if hasattr(patcher, "model") and hasattr(patcher.model, "named_modules"):
                yield from patcher.model.named_modules()

        def materialize_int8_lora_patches(patcher):
            """Same as INT8ModelSave: bake non-dynamic INT8 LoRA low-vram
            functions into weights before we bypass LazyCastingParam."""
            if _is_dynamic_lora_enabled() or not hasattr(patcher, "patch_weight_to_device"):
                return

            patches = getattr(patcher, "patches", None)
            if not patches:
                return

            load_device = getattr(patcher, "load_device", None)
            materialized = 0
            for name, module in iter_model_modules(patcher):
                if not getattr(module, "_is_quantized", False):
                    continue

                weight_key = name + ".weight" if name else "weight"
                if weight_key not in patches:
                    continue

                try:
                    current_weight = getattr(module, "weight", None)
                    device_to = load_device if load_device is not None else getattr(current_weight, "device", None)
                    patcher.patch_weight_to_device(weight_key, device_to=device_to)
                    if hasattr(module, "weight_lowvram_function"):
                        module.weight_lowvram_function = None
                    materialized += 1
                except Exception as e:
                    logging.warning(
                        f"INT8 CLIP Save: failed to materialize LoRA patch for {weight_key}: {e}. "
                        "The saved checkpoint may miss this LoRA patch."
                    )

            if materialized > 0:
                logging.info(f"INT8 CLIP Save: materialized {materialized} INT8 LoRA patched weight(s) before saving.")

        # Finalize any deferred INT8 layers, same as the model save path.
        finalize_fn = getattr(model_patcher, "finalize_pending_int8", None)
        if finalize_fn is not None:
            finalize_fn()

        # Force-load onto the compute device so every layer's int8 weight is
        # observable, same reasoning as INT8ModelSave. clip.load_model() alone
        # does NOT force_full_load, so lowvram setups could otherwise leave
        # some layers unmaterialized.
        try:
            comfy.model_management.load_models_gpu([model_patcher], force_full_load=True)
        except Exception as e:
            logging.warning(
                f"INT8 CLIP Save: full-load pre-pass failed ({e}); falling back to "
                "default load_models_gpu without force_full_load."
            )
            try:
                comfy.model_management.load_models_gpu([model_patcher])
            except Exception as e2:
                logging.warning(
                    f"INT8 CLIP Save: load_models_gpu fallback also failed ({e2}); "
                    "continuing best-effort. The saved checkpoint may be "
                    "incomplete if LoRA patches were not applied."
                )

        if finalize_fn is not None:
            finalize_fn()

        materialize_int8_lora_patches(model_patcher)

        # Collect comfy_quant / weight_scale extra_keys. NOTE: unlike
        # INT8ModelSave, no "model." prefix here -- CLIP.state_dict_for_saving()
        # calls patcher.model_state_dict_for_saving() with prefix="" (default),
        # so keys must match the raw named_modules() dotted path exactly, since
        # that's what will later get split/stripped per-encoder below.
        for name, module in iter_model_modules(model_patcher):
            if module_has_int8_param(module):
                mark_module_for_direct_save(module)

            if getattr(module, "_is_quantized", False):
                use_convrot = bool(getattr(module, "_use_convrot", False))
                quant_conf = {"format": "int8_tensorwise", "convrot": use_convrot}
                if use_convrot:
                    try:
                        from .int8_quant import CONVROT_GROUP_SIZE
                    except Exception:
                        CONVROT_GROUP_SIZE = 256
                    quant_conf["convrot_groupsize"] = int(
                        getattr(module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                    )
                quant_conf["per_row"] = bool(getattr(module, "_is_per_row", False))

                prefix = name + "." if name else ""

                extra_keys[prefix + "comfy_quant"] = torch.tensor(
                    list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8
                )

                if getattr(module, "_weight_scale_scalar", None) is not None:
                    extra_keys[prefix + "weight_scale"] = torch.tensor(module._weight_scale_scalar)

                mark_module_for_direct_save(module)

        original_lazy_new = comfy.model_patcher.LazyCastingParam.__new__
        original_lazy_piece_new = comfy.model_patcher.LazyCastingParamPiece.__new__

        def lazy_casting_param_new(cls, model, key, tensor):
            requires_grad = tensor.is_floating_point() or tensor.is_complex()
            return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

        def lazy_casting_param_piece_new(cls, caster, state_dict_key, tensor):
            requires_grad = tensor.is_floating_point() or tensor.is_complex()
            return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

        try:
            comfy.model_patcher.LazyCastingParam.__new__ = staticmethod(lazy_casting_param_new)
            comfy.model_patcher.LazyCastingParamPiece.__new__ = staticmethod(lazy_casting_param_piece_new)

            clip.load_model()
            clip_sd = clip.state_dict_for_saving()
            for k in extra_keys:
                clip_sd[k] = extra_keys[k]
        finally:
            comfy.model_patcher.LazyCastingParam.__new__ = original_lazy_new
            comfy.model_patcher.LazyCastingParamPiece.__new__ = original_lazy_piece_new
            for module, had_flag, old_flag in patched_modules:
                if had_flag:
                    module.comfy_patched_weights = old_flag
                else:
                    try:
                        delattr(module, "comfy_patched_weights")
                    except AttributeError:
                        pass

        # Same per-encoder split/strip as stock CLIPSave, EXCEPT the prefix
        # list is derived dynamically from the model's actual top-level child
        # modules instead of a hardcoded list. Stock CLIPSave's hardcoded list
        # (clip_l./t5xxl./gemma2_2b./etc.) predates newer single-encoder
        # architectures like Krea2/Boogu/Ideogram4/Ovis (all SD1ClipModel
        # wrapping a Qwen3-VL variant under an attribute named e.g.
        # "qwen3vl_4b"). Missing that prefix means it never gets stripped,
        # detect_te_model() on reload can't recognize the resulting keys, and
        # comfy silently falls back to a default CLIP_L -- which is exactly
        # what produced the 768-dim garbage conditioning here.
        try:
            child_prefixes = [f"{n}." for n, _ in clip.cond_stage_model.named_children()]
        except Exception:
            child_prefixes = []
        # Keep the legacy list too (belt-and-suspenders for any child name
        # collisions/edge cases) plus the catch-all "" bucket last.
        legacy_prefixes = ["clip_l.", "clip_g.", "clip_h.", "t5xxl.", "pile_t5xl.", "mt5xl.",
                           "umt5xxl.", "t5base.", "gemma2_2b.", "llama.", "hydit_clip."]
        seen_prefixes = set()
        ordered_prefixes = []
        for p in child_prefixes + legacy_prefixes + [""]:
            if p not in seen_prefixes:
                seen_prefixes.add(p)
                ordered_prefixes.append(p)

        for prefix in ordered_prefixes:
            k = list(filter(lambda a: a.startswith(prefix), clip_sd.keys()))
            current_clip_sd = {}
            for x in k:
                current_clip_sd[x] = clip_sd.pop(x)
            if len(current_clip_sd) == 0:
                continue

            p = prefix[:-1]
            replace_prefix = {}
            filename_prefix_ = filename_prefix
            if len(p) > 0:
                filename_prefix_ = "{}_{}".format(filename_prefix_, p)
                replace_prefix[prefix] = ""
            replace_prefix["transformer."] = ""

            full_output_folder, filename, counter, subfolder, filename_prefix_ = folder_paths.get_save_image_path(filename_prefix_, self.output_dir)

            output_checkpoint = f"{filename}_{counter:05}_.safetensors"
            output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

            current_clip_sd = comfy.utils.state_dict_prefix_replace(current_clip_sd, replace_prefix)

            # Reverse comfy.sd's load-time remap for Qwen3-VL-family text
            # encoders (Krea2/Boogu/Ideogram4/plain Qwen3VL). At load,
            # load_text_encoder_state_dicts() does:
            #   state_dict_prefix_replace(sd, {"model.language_model.": "model.",
            #                                   "model.visual.": "visual.",
            #                                   "lm_head.": "model.lm_head."})
            # on the RAW on-disk keys before the module is even built. That
            # rename is irreversible once the module is instantiated -- by the
            # time we read state_dict() back off the live module, the HF-native
            # "model.language_model."/"model.visual." names are gone, replaced
            # by comfy's internal "model."/"visual." names. Writing those out
            # as-is is what produced model.layers.* / bare visual.* instead of
            # model.language_model.layers.* / model.visual.* -- detect_te_model()
            # doesn't recognize that shape and silently falls back to CLIP_L.
            #
            # Gate strictly on a bare top-level "visual." group: that's the one
            # unambiguous fingerprint of this specific remap having happened
            # (no non-VL comfy text encoder ever produces a bare "visual."
            # group), so this won't touch clip_l/t5xxl/gemma2_2b/llama/etc.
            has_bare_visual = any(kk.startswith("visual.") for kk in current_clip_sd)
            if has_bare_visual:
                reverse_remap = {}
                if any(kk.startswith("model.lm_head.") for kk in current_clip_sd):
                    reverse_remap["model.lm_head."] = "lm_head."
                # Generic "model." -> "model.language_model." MUST run before the
                # "visual." rule below. state_dict_prefix_replace rescans live
                # dict keys per rp, so if "visual." -> "model.visual." ran first,
                # its own output would then get re-caught by this "model." rule
                # on the next iteration and become "model.language_model.visual.*".
                reverse_remap["model."] = "model.language_model."
                reverse_remap["visual."] = "model.visual."
                current_clip_sd = comfy.utils.state_dict_prefix_replace(current_clip_sd, reverse_remap)

            for kk in current_clip_sd:
                t = current_clip_sd[kk]
                if isinstance(t, torch.Tensor) and not t.is_contiguous():
                    current_clip_sd[kk] = t.contiguous()

            comfy.utils.save_torch_file(current_clip_sd, output_checkpoint, metadata=metadata)

        return {}
