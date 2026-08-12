class ModelManager:

    _models = {}
    _tokenizers = {}

    @classmethod
    def is_model_loaded(cls, model_name):

        return model_name in cls._models

    @classmethod
    def register_model(cls, model_name, model, tokenizer):

        if model_name not in cls._models:
            cls._models[model_name] = model
            cls._tokenizers[model_name] = tokenizer
            print(f"Model {model_name} registered successfully.")

    @classmethod
    def get_model(cls, model_name):

        if model_name not in cls._models:
            raise ValueError(f"Model {model_name} has not been loaded yet. Please load it first.")
        return cls._models[model_name], cls._tokenizers[model_name]
