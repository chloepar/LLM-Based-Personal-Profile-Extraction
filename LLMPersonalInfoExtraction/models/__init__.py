from .GPT import GPT
from .Gemini import Gemini
from .Llama import Llama
from .Groq import Groq


def create_model(config):
    provider = config["model_info"]["provider"].lower()
    if provider == 'gpt':
        model = GPT(config)
    elif provider == 'gemini':
        model = Gemini(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'groq':
        model = Groq(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model