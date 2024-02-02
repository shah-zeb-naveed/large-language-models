# Large Language Models Repository

This repository encompasses various projects related to large language modeling, focusing on different tasks. Below is an overview of the projects and their associated information.

## Code Parrot: Training an LLM for Python Code Completion Assistance

- **Overview:** This project involves fine-tuning a GPT2-based model for code completion assistance in Python.

### HuggingFace Hub
| Demo on Spaces                                                        | Model Card                                              |
|-----------------------------------------------------------------------|---------------------------------------------------------|
| [Demo on Spaces](https://huggingface.co/spaces/shahzebnaveed/shahzebnaveed-codeparrot-ds) (Example Input: `import matplotlib.pyplot`) | [Model Card](https://huggingface.co/shahzebnaveed/codeparrot-ds) |

- **Training:** The GPT2 network is trained from scratch using Causal Language Modeling in PyTorch and HuggingFace Transformers.
- **Azure Deployment:** Due to limited native support in Azure for MLFlow format with Transformers, a CUSTOM_MODEL with a custom scoring script is used. The solution leverages MLFlow PyFunc flavour to implement a core class.

## Fine-Tuning DistilBert for Masked Language Modelling on IMDB Dataset

### Model Card and Demo Inference Endpoint
| Model Card and Demo                                                                                 |
|------------------------------------------------------------------------------------------------------|
| [Model Card and Demo](https://huggingface.co/shahzebnaveed/distilbert-base-uncased-finetuned-imdb) |

## Fine-Tuning BERT-base-cased for Extractive Question Answering

### Model Card and Demo Inference Endpoint
| Model Card and Demo                                                                             |
|----------------------------------------------------------------------------------------------|
| [Model Card and Demo](https://huggingface.co/shahzebnaveed/bert-finetuned-squad?context=My+name+is+Clara+and+I+live+in+Berkeley.&text=What%27s+my+name%3F) |

## Fine-Tuning Helsinki-NLP/opus-mt-en-fr for English to French Translation

### Model Card and Demo Inference Endpoint
| Model Card and Demo                                                                         |
|---------------------------------------------------------------------------------------------|
| [Model Card and Demo](https://huggingface.co/shahzebnaveed/marian-finetuned-kde4-en-to-fr) |

**Credits:** Code submissions adapted for the HuggingFace NLP course.
