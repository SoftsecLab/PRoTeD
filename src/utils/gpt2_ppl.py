# python -m src.utils.gpt2_ppl
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer ,GPT2Config
from src.utils.manager import ModelManager
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
_tokenizer = None
_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_llscore_ppl(text: str, model_path="./models/gpt2-medium"):
    global _tokenizer, _model


    if _tokenizer is None or _model is None:
        _tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        _model = GPT2LMHeadModel.from_pretrained(model_path)
        _tokenizer.pad_token = _tokenizer.eos_token
        _model.to(device).eval()


    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss


    seq_len = inputs["input_ids"].shape[1]
    llscore = -loss.item() * seq_len
    ppl = torch.exp(loss).item()

    return llscore, ppl
class GPT2PPLCalculator:
    def __init__(self, model_name="./models/gpt2-medium"):
        import warnings
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


        _original_forward = GPT2LMHeadModel.forward

        def patched_forward(self, *args, **kwargs):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _original_forward(self, *args, **kwargs)

        GPT2LMHeadModel.forward = patched_forward

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if ModelManager.is_model_loaded(model_name):
            self.model, self.tokenizer = ModelManager.get_model(model_name)
        else:
            print(f"Loading GPT-2 model: {model_name} ...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.loss_type = "ForCausalLMLoss"
            self.model.eval()
            self.model.to(self.device)
            ModelManager.register_model(model_name, self.model, self.tokenizer)


    def compute_llscore_ppl(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.config.loss_type = "ForCausalLMLoss"
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
        perplexity = torch.exp(outputs.loss).item()


        return log_likelihood, perplexity

if __name__ == "__main__":
    import warnings
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


    _original_forward = GPT2LMHeadModel.forward

    def patched_forward(self, *args, **kwargs):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _original_forward(self, *args, **kwargs)

    GPT2LMHeadModel.forward = patched_forward
    calculator = GPT2PPLCalculator()
    calculator.model.config.loss_type = "ForCausalLMLoss"
    test_text = "This is a simple test sentence."
    llscore, ppl = calculator.compute_llscore_ppl(test_text)
    print(f"LLScore: {llscore}, PPL: {ppl}")
