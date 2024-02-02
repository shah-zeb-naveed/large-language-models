import mlflow.pyfunc
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeParrot(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = AutoModelForCausalLM.from_pretrained(context.artifacts["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["model"])

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(inputs["input_ids"])
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs]