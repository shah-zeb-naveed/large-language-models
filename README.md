# Large Language Models Repository

This repository hosts various projects on large language modeling, focusing on diverse tasks with advanced techniques. Below is a concise overview:

## 🚀 Top Models on 🤗 Hugging Face Hub [Profile Link ↗️](https://huggingface.co/shahzebnaveed)
- [shahzebnaveed/NeuralHermes-2.5-Mistral-7B](https://huggingface.co/shahzebnaveed/NeuralHermes-2.5-Mistral-7B) 🔥 2k downloads
- [shahzebnaveed/StarlingHermes-2.5-Mistral-7B-slerp](https://huggingface.co/shahzebnaveed/StarlingHermes-2.5-Mistral-7B-slerp) 🔥 2k downloads

## Causal Language Modelling
- Fine-tuning a GPT2-based model for Python code completion (Code Parrot)
    - [Model Link](https://huggingface.co/shahzebnaveed/codeparrot-ds)
- **Training:** GPT2 network trained from scratch using PyTorch and HuggingFace Transformers.
- **Azure Deployment:** Custom model with MLFlow PyFunc flavour due to Azure's limited native support.

## Reinforcement Learning from Human Feedback
- Direct Preference Optimization on Mistral with Orca Dataset using QLoRa
    - [Model Link](https://huggingface.co/shahzebnaveed/StarlingHermes-2.5-Mistral-7B-slerp)
- Proximal Policy Optimization on GPT-2 to control sentiment of movie reviews using BERT as a reward model 

## Retrieval Augmented Generation on Databricks
- Using Databricks's built-in vector store and retrieval to implement a RAG Chatbot
- Indexed data and deployed model to Databricks Models endpoint

## Instruction Tuning
- Instruction fine-tuning LLAMA 2 using Alpaca dataset

## Optimization Techniques
- Quantization (Bitsandbytes)
- PEFT (LoRa, QLoRa)
- llama.cpp for GGUF format
    - [Model Link](https://huggingface.co/shahzebnaveed/EvolCodeLlama-7b-GGUF1)

## Mixture-of-Experts
- Fine-tuning google's switch base transformers for summarization task
- [Model Link](https://huggingface.co/shahzebnaveed/moe_switch_transformer_summarization)

## Model Merging
- Merge-kit tool to create different merged models
- [Model Link](https://huggingface.co/shahzebnaveed/StarlingHermes-2.5-Mistral-7B-slerp)

## Prompt Tuning
- Fine-tuning Bloomz model using causal language modelling
    - [Model Link](https://huggingface.co/shahzebnaveed/bloomz-560m_prompt_tuning_clm)
- Fine-tuning T5 for prefix tuning using sequence-to-sequence
    - [Model Link](https://huggingface.co/shahzebnaveed/t5-large_PREFIX_TUNING_SEQ2SEQ)

## Langchain Chains
- ReAct agents with custom tool
- RAG with chat memory (Chroma and Pinecone)
- LangServe & Fast API to deploy RAG chatbot

## Multi-Agent Chats
- Python verification chat with Autogen where one agent executes the code

## Masked Language Modelling
- Fine-tuning DistilBert on IMDB Dataset
    - [Model Link](https://huggingface.co/shahzebnaveed/distilbert-base-uncased-finetuned-imdb)

## Extractive Question Answering
- Fine-tuning BERT-base-cased
    - [Model Link](https://huggingface.co/shahzebnaveed/bert-finetuned-squad?context=My+name+is+Clara+and+I+live+in+Berkeley.&text=What%27s+my+name%3F)

## Translation
- Fine-tuning Helsinki-NLP/opus-mt-en-fr for English to French 
    - [Model Link](https://huggingface.co/shahzebnaveed/marian-finetuned-kde4-en-to-fr)

## Token Classification
- Fine-tuning roBERTa using LoRa (Low Rank Adaptation)
    - [Model Link](https://huggingface.co/shahzebnaveed/roberta-large-lora-token-cls)

# Notes on Hugging Face Transformers

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


**Credits:**
- Adapted code submissions from HuggingFace NLP course.
- Blog posts/tutorials from [llm-course](https://github.com/mlabonne/llm-course).
- Databricks tutorial on RAG
- Merge-kit tool
