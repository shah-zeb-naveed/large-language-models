# codeparrot
Training an LLM for Python Code Completion Assistance

HuggingFace Hub:

Demo on Spaces: https://huggingface.co/spaces/shahzebnaveed/shahzebnaveed-codeparrot-ds (Example Input: import matplotlib.pyplot)
Model Card: https://huggingface.co/shahzebnaveed/codeparrot-ds

Training:

A GPT2 network is trained from scratch using Causal Language Modelling in PyTorch and HuggingFace Transformers.

Azure Deployment:

Although MLFlow supports Transformers, the native support in Azure for MLFlow format is limited. Therefore, a CUSTOM_MODEL with a custom scoring script has to be specified. This solution uses MLFlow PyFunc flavour to implement a core class. 
