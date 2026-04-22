from groq import Groq as GroqClient

from .Model import Model


class Groq(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.api_key = api_keys[api_pos]
        self.set_API_key()
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
    def set_API_key(self):
        self.client = GroqClient(api_key=self.api_key)
        
    def query(self, msg, image_path=None):
        assert image_path is None, f"Groq LLaMA model does not support image input"
        
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "user", "content": msg}
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens
        )
        response = completion.choices[0].message.content
        return response
