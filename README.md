# large-language-models

This repository contains various projects around large language modeling including:

- fine-tuning for masked language modeling using distilbert.
- from the scratch training of GPT2 architecture for causal language modeling for python code assistance (code parrot)
- fine-tuning distilbert for translation task: english to french translation
- fine-tuning BERT for extraction question-answering



**Code Parrot: Training an LLM for Python Code Completion Assistance**

HuggingFace Hub:

Demo on Spaces: https://huggingface.co/spaces/shahzebnaveed/shahzebnaveed-codeparrot-ds (Example Input: import matplotlib.pyplot)
Model Card: https://huggingface.co/shahzebnaveed/codeparrot-ds

Training:

A GPT2 network is trained from scratch using Causal Language Modelling in PyTorch and HuggingFace Transformers.

Azure Deployment:

Although MLFlow supports Transformers, the native support in Azure for MLFlow format is limited. Therefore, a CUSTOM_MODEL with a custom scoring script has to be specified. This solution uses MLFlow PyFunc flavour to implement a core class.

**Fine-Tuning DistilBert for Masked Language Modelling on IMDB dataset**

Model Card and Demo Inference Endpoint:  https://huggingface.co/shahzebnaveed/distilbert-base-uncased-finetuned-imdb

**Fine-Tuning BERT-base-cased for extractive question answering**

Model Card and Demo Inference Endpoint: https://huggingface.co/shahzebnaveed/bert-finetuned-squad?context=My+name+is+Clara+and+I+live+in+Berkeley.&text=What%27s+my+name%3F

**Fine-tuning Helsinki-NLP/opus-mt-en-fr for English to French Translation**

ModelModel Card and Demo Inference Endpoint: https://huggingface.co/shahzebnaveed/marian-finetuned-kde4-en-to-fr

**Credits**

Code submissions adapted for the HuggingFace NLP course.
