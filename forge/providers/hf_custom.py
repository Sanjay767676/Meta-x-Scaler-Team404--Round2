"""Hugging Face base model + LoRA adapter (PEFT)."""

from __future__ import annotations

import logging
from typing import Any

from forge.llm_types import LLMResponse

logger = logging.getLogger("forge.hf_custom")


class HFCustomProvider:
    name = "custom_hf"

    def __init__(
        self,
        base_model_id: str,
        adapter_id: str,
        hf_token: str | None,
        max_new_tokens: int = 512,
    ) -> None:
        self.base_model_id = base_model_id
        self.adapter_id = adapter_id
        self.hf_token = hf_token or None
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._load_error is not None:
            return
        if self._model is not None:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok_kw: dict[str, Any] = {"trust_remote_code": True}
            if self.hf_token:
                tok_kw["token"] = self.hf_token
            logger.info("[custom_hf] loading tokenizer %s", self.base_model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, **tok_kw)
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else None
            model_kw: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }
            if self.hf_token:
                model_kw["token"] = self.hf_token
            if device_map:
                model_kw["device_map"] = device_map

            logger.info("[custom_hf] loading base %s", self.base_model_id)
            base = AutoModelForCausalLM.from_pretrained(self.base_model_id, **model_kw)
            logger.info("[custom_hf] loading adapter %s", self.adapter_id)
            peft_kw: dict[str, Any] = {}
            if self.hf_token:
                peft_kw["token"] = self.hf_token
            self._model = PeftModel.from_pretrained(base, self.adapter_id, **peft_kw)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[custom_hf] load failed: %s", self._load_error)

    def generate(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        import torch

        self._load()
        if self._load_error or self._model is None or self._tokenizer is None:
            raise RuntimeError(self._load_error or "HF custom model not loaded")

        messages = [
            {"role": "system", "content": system_prompt or "You are a Python coding assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = f"{system_prompt}\n\n{prompt}\nAssistant:\n"

        inputs = self._tokenizer(text, return_tensors="pt", return_attention_mask=True)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        from transformers import GenerationConfig

        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            out = self._model.generate(**inputs, generation_config=gen_cfg)
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        decoded = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not decoded:
            raise ValueError("HF custom returned empty text")
        logger.info("[custom_hf] generated %d chars", len(decoded))
        return LLMResponse(
            provider=self.name,
            model=f"{self.base_model_id}+{self.adapter_id}",
            content=decoded,
            raw={"status": "ok"},
        )
