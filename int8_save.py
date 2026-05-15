import folder_paths
import comfy.sd
import comfy.model_patcher
import json
import os
import torch
from comfy.cli_args import args

class INT8ModelSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "int8_models/INT8_Model"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "loaders"

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        # if not args.disable_metadata:
        #     metadata["prompt"] = prompt_info
        #     if extra_pnginfo is not None:
        #         for x in extra_pnginfo:
        #             metadata[x] = json.dumps(extra_pnginfo[x])

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

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

        def iter_model_modules(model_patcher):
            roots = []
            if hasattr(model_patcher, "model"):
                roots.append(model_patcher.model)
                diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
                if diffusion_model is not None:
                    roots.append(diffusion_model)

            seen_roots = set()
            for root in roots:
                root_id = id(root)
                if root_id in seen_roots or not hasattr(root, "named_modules"):
                    continue
                seen_roots.add(root_id)
                yield from root.named_modules()
        
        # Finalize any deferred INT8 layers (Aimdo/Windows deferred-load path sets
        # _pending_int8_finalize instead of quantizing immediately). Without this,
        # those modules still have _is_quantized=False at save time and no
        # comfy_quant keys are emitted.
        finalize_fn = getattr(model, "finalize_pending_int8", None)
        if finalize_fn is not None:
            finalize_fn()

        # We need to peek at the model's actual modules to save comfy_quant and weight_scale
        if hasattr(model, "model"):
            for name, module in iter_model_modules(model):
                if module_has_int8_param(module):
                    # ComfyUI's LazyCastingParam subclasses torch.nn.Parameter
                    # with requires_grad=True by default, which is invalid for
                    # int8 tensors. Mark all int8 modules for direct save.
                    mark_module_for_direct_save(module)

                if getattr(module, "_is_quantized", False):
                    # 1. Comfy Quant Hint
                    quant_conf = {"convrot": getattr(module, "_use_convrot", False)}
                    if hasattr(module, "_convrot_groupsize"):
                        quant_conf["convrot_groupsize"] = module._convrot_groupsize
                        
                    # Prepend 'model.' as comfy.sd.save_checkpoint typically adds this to all weights
                    # but may not add it to extra_keys. This ensures they stay alongside weights.
                    prefix = "model." + name + "." if name else "model."
                    
                    extra_keys[prefix + "comfy_quant"] = torch.tensor(
                        list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8
                    )
                    
                    # 2. Handle scalar weight_scale which is not registered as a buffer
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
            comfy.sd.save_checkpoint(output_checkpoint, model, metadata=metadata, extra_keys=extra_keys)
        finally:
            comfy.model_patcher.LazyCastingParam.__new__ = original_lazy_new
            comfy.model_patcher.LazyCastingParamPiece.__new__ = original_lazy_piece_new
            # Restore module states so we don't break dynamic VRAM management
            for module, had_flag, old_flag in patched_modules:
                if had_flag:
                    module.comfy_patched_weights = old_flag
                else:
                    delattr(module, "comfy_patched_weights")
                    
        return {}
