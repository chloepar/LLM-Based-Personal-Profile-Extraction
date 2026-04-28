import re
import time

from groq import Groq as GroqClient
from groq import RateLimitError

from .Model import Model


class Groq(Model):
    def __init__(self, config):
        super().__init__(config)
        self.api_keys = config["api_key_info"]["api_keys"]
        assert len(self.api_keys) > 0, "Please provide at least one API key"
        self.api_key_index = int(config["api_key_info"]["api_key_use"])
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = GroqClient(api_key=self.api_keys[self.api_key_index])

    def _rotate_key(self):
        """Switch to the next API key, returns True if we wrapped around (all keys tried)."""
        next_index = (self.api_key_index + 1) % len(self.api_keys)
        wrapped = next_index <= self.api_key_index and next_index == 0
        self.api_key_index = next_index
        self.client = GroqClient(api_key=self.api_keys[self.api_key_index])
        return wrapped

    def set_API_key(self):
        self.client = GroqClient(api_key=self.api_keys[self.api_key_index])

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

        keys_tried = 0
        max_attempts = len(self.api_keys) * 4

        for attempt in range(max_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=[{"role": "user", "content": msg}],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens
                )
                return completion.choices[0].message.content
            except RateLimitError as e:
                keys_tried += 1
                if keys_tried < len(self.api_keys):
                    self._rotate_key()
                    print(f"[Groq] Rate limit hit. Rotating to key {self.api_key_index + 1}/{len(self.api_keys)}...")
                else:
                    # All keys exhausted — wait for the slowest key's suggested time
                    wait = self._parse_wait_seconds(str(e))
                    print(f"[Groq] All {len(self.api_keys)} key(s) rate limited. Waiting {wait:.0f}s then retrying (round {attempt // len(self.api_keys) + 1})...")
                    time.sleep(wait)
                    keys_tried = 0
                    self._rotate_key()

        raise RuntimeError("Groq rate limit exceeded after all retries")
