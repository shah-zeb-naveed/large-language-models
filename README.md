# codeparrot
Training GPT2 for Causal Language Modelling using PyTorch and HuggingFace Transformers

Although MLFlow supports Transformers, the native support in Azure for MLFlow format is limited. Therefore, a CUSTOM_MODEL with a custom scoring script has to be specified. This solution uses MLFlow PyFunc flavour to implement a core class. 
