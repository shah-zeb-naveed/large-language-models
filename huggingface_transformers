# Some Notes on Hugging Face Transformers

## 8-bit Quantization
- **LLM.int8()**: 8-bit matrix multiplication with an outlier threshold (outliers in hidden states processed in fp16)
  - **llm_int8_has_fp16_weight**: Runs int8() with fp16 main weights.
    - `has_fp16_weights` must be set to False to load weights in int8 along with quantization statistics.
    - Set to True for mixed int8/fp16 training.
  - **Usage**: Inference only, fine-tuning, or pre-training?
    - Likely not pre-training. Pure 8-bit or 4-bit training is not possible, but training adapters on top is feasible.

## 4-bit Quantization (QLORA)
- **Weights Quantization**: fp4/nf4 as proposed in QLORA.
  - In plain LoRA, both transformer and adapters are 16 bits; in QLORA, the adapter is 16 bits and the transformer is 4 bits.
  - Optimizer states are 32 bits (paged to CPU in QLORA).
  - From the paper: "QLORA has one low-precision storage data type (usually 4-bit) and one computation data type (usually BFloat16). QLORA weight tensors are dequantized to BFloat16 for matrix multiplication."
  - **Fine-tuning Approach**.
  - **bnb_4bit_compute_dtype**: Sets compute type, which may differ from the input type.
  - **4bit_double_quantization**: Constants quantized.
  - Separate storage and computation data types for accuracy and stability.

## Accelerate
- **Training**: Pure 4-bit or 8-bit training is not possible.
  - Use PEFT (e.g., LoRA adapters) and train on top.
  - Cannot add adapters on top of a quantized model.
  - Can fine-tune a quantized model with the support of adapters.
  - Example from Llama 2 demo: 
    - Load a quantized 4-bit model.
    - Prepare for k-bit training (dequantize some layers).
    - Add LoRA adapters (add matrix decomposition deltas).
    - Training:
      - FP16: Mixed precision (LoRA adapters in fp16/32, others in 4-bit).
      - 8-bit Adam optimizer (optimizer states like squared sum (Adam) or running gradients).

## Optimizer States
- Typically 32-bit, but bitsandbytes offers 8-bit Adam, etc.
- Stateful optimizers maintain gradient statistics over time (e.g., SGD with momentum or Adam).

## Mixed Precision
- **torch.cuda.amp.autocast**: 
  - "Some operations (e.g., linear layers, convolutions) are faster in lower precision. Others (e.g., reductions) require float32 for dynamic range. Mixed precision matches each operation to its appropriate datatype."
  - Mixed precision switches between 32-bit and 16-bit operations during training, enhancing training speed and reducing memory usage.
  - FP16 cuts memory in half and speeds up training.
  - Mixed precision uses FP32 weights as a reference, with FP16/BF16 computation during forward and backward passes. Gradients update FP32 main weights.
  - Applicable during inference for the forward pass as well.

## Quantization
- Reduces memory and computational costs by using lower-precision data types (e.g., int8).
  - Techniques include range mapping/scaling, shifting, clipping, etc.
  - Types:
    - Post-training vs. Quantization-Aware (pre-training/fine-tuning, e.g., QLORA).
    - No pre-training references found, only fine-tuning.
  - Enhances model performance during inference beyond lower floating point and converts to integer.
  - Int8() aims to halve memory usage compared to 16-bit. "Mixed precision" in quantization treats outliers with 16-bit computation.
    - Input must be FP16.

## Compute Data Types: BF16 vs. FP16
- **FP16**: Some operations and parameters stay in 32-bit, others convert to 16-bit and back (training and inference).
- **4bit_compute_dtype = BF16**: Load quantized, store in 4-bit, and compute in 16-bit (frozen).
  - Adapters can be trained in FP16 mixed precision.
