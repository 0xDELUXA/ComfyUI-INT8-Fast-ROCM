# Comfy INT8 Acceleration

This node speeds up Flux2, Chroma, Z-Image, Ernie Image in ComfyUI by using INT8 quantization, delivering between 1.5~2x faster inference on my 3090 depending on the model. It should work on any NVIDIA GPU with enough INT8 TOPS. It appears to be faster than FP8 on 40-Series and above as well. 
Works with lora, torch compile.

---

Updates:

2026-15-05:

Bringing back stochastic lora. Some loras appear to need it, others don't, try it if your lora is not working and you don't like pre-lora. See our measurements [here](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/RAMExp/Metrics.md#some-loras-require-stochastic-lora-to-work).

Attempt at reducing RAM usage

Fixed an issue with Pre-Lora crashing on windows

2026-10-05:

Overhauled the entire lora system. Normal lora loader node works now, no need for specialized lora loaders.

Converted QuaRot to ConvRot, which is a small but free quality gain.

Added Pre-Lora node, which you can connect to the INT8 Model loader to merge loras before utilizing on the fly quantization. 

For more info on quality of convrot, lora approaches see the [Metrics](Metrics.md)

---

Pre-quantized checkpoints were recommended for most architectures, but on-the-fly quantization with ConvRot is better in all cases.
However, ConvRot is also a little slower, so these prequantized models are still useful. Avoid using INT8 Tensorwise models.

**Shoutout to [vistralis](https://huggingface.co/vistralis) for these:** 

| Model | Link |
|-------|------|
| FLUX.2-klein-base-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-9b-INT8-transformer) |
| FLUX.2-klein-base-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-4b-INT8-transformer) |
| FLUX.2-klein-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-9b-INT8-transformer) |
| FLUX.2-klein-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer) |

**My own:**

| Model | Link |
|-------|------|
| Chroma1-HD² | ~~[Download](https://huggingface.co/bertbobson/Chroma1-HD-INT8Tensorwise)~~ |
| Z-Image-Base¹ | ~~[Download](https://huggingface.co/bertbobson/Z-Image-Base-INT8-QUIP)~~ 
| Z-Image-Turbo² | ~~[Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise)~~ |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |

¹Z-Image Base weights have been Deprecated in favor of Convrot OTF, which is higher quality.

²Tensorwise models are worse than on the fly quantization since we switched to row-wise INT8


# Speed:

Measured on a 3090 at 1024x1024, 26 steps with Flux2 Klein Base 9B.

| Format | Speed (s/it) ↓ | Relative Speedup |
|-------|--------------|------------------|
| bf16 | 2.07 | 1.00× |
| bf16 compile | 2.24 | 0.92× |
| fp8 | 2.06 | 1.00× |
| int8 | 1.64 | 1.26× |
| int8 compile ★| 1.04 | 1.99× |
| gguf8_0 compile | 2.03 | 1.02× |

3090, Qwen Image 2512.

| Format | Speed (s/it) ↓ |
|-------|--------------|
| Nunchaku INT4 Best Quality | 1.21 |
| Nunchaku INT4 with R128 Lora | 1.36 |
| INT8 ConvRot compile | 1.26 |
| INT8 Row compile ★| 1.18 |
| INT8 R128 Lora | No slowdown, except if dynamic. |

I would also like to point out that we beat Nunchaku INT4 on every quality measurement in the [Quality Metrics](Metrics.md)

Additionally, the quality of loras applied with [this nunchaku lora node](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader) appears to be degraded.

Klein 9B, Measured on an 8gb 5060, same settings as the 3090 run:

| Format | Speed (s/it) ↓ | Relative Speedup |
|-------|--------------|------------------|
| fp8 | 3.04 | 1.00× |
| fp8 fast | 3.00 | 1.00× |
| fp8 compile | couldn't get to work | ??× |
| int8 | 2.53 | 1.20× |
| int8 compile ★| 2.25 | 1.35× |

8gb RTX 5060, Anima, Comfy version from 2026-05-02, Pytorch 2.11+CU13.0, latest kitchen triton and everything else

| Format | Speed (it/s) ↑ |
|-------|--------------|
| bf16 | 0.78 |
| INT8 ConvRot | 1.12 |
| INT8 Row | 1.24 |
| INT8 ConvRot Compile | 1.47 |
| MXFP8 | 0.89 |
| MXFP8 --fast | 0.93 |
| MXFP8 + Compile | Still failing. |

Finally have gotten compile with --fast to work with mxfp8, PyTorch 2.13.0.dev20260511+cu132, RTX5060 same as before.

Quality results for this run, can be found here: [Anima Results](Metrics.md#anima-on-a-5060)

| Format | Speed (it/s) ↑ |
|-------|--------------|
| MXFP8 --fast + Compile | 1.37it |
| INT8 ConvRot + Compile | 1.47it |


# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!

## newgrit1004 for the ConvRot code I basically copied
https://github.com/newgrit1004/ComfyUI-ZImage-Triton

## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of a day
