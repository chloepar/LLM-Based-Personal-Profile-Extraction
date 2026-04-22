import re
import time

from groq import Groq as GroqClient
from groq import RateLimitError

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

    def _parse_wait_seconds(self, error_msg):
        """Parse the suggested retry delay from a Groq rate limit error message."""
        m = re.search(r'in (\d+)m(\d+\.?\d*)s', error_msg)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2)) + 5
        m = re.search(r'in (\d+\.?\d*)s', error_msg)
        if m:
            return float(m.group(1)) + 5
        return 60  # fallback

    def query(self, msg, image_path=None):
        assert image_path is None, f"Groq LLaMA model does not support image input"

        # llama-3.1-8b-instant: 6,000 TPM hard limit per request.
        # Reserve 150 tokens for output → 5,850 input tokens max.
        # ~4 chars/token → 5,850 * 4 = 23,400; use 22,000 for safety margin.
        max_chars = 22_000
        if len(msg) > max_chars:
            msg = msg[:max_chars]

        for attempt in range(4):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=[{"role": "user", "content": msg}],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens
                )
                return completion.choices[0].message.content
            except RateLimitError as e:
                wait = self._parse_wait_seconds(str(e))
                print(f"[Groq] Rate limit hit. Waiting {wait:.0f}s then retrying (attempt {attempt + 1}/4)...")
                time.sleep(wait)

        raise RuntimeError("Groq rate limit exceeded after 4 retries")
