from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from config import CONFIG, AgentConfig, LLMConfig
from intent_classifier import IntentClassifier
from model_manager import manager as model_manager


class LLM:
    def __init__(self, config: Optional[LLMConfig] = None):
        cfg = config or CONFIG.llm

        self.model_name = cfg.model_name
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k
        self.repetition_penalty = cfg.repetition_penalty
        self.enable_thinking = cfg.enable_thinking
        self._torch_dtype = cfg.torch_dtype
        self._device_map = cfg.device_map
        self._load_in_4bit = cfg.load_in_4bit

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy loading — model is loaded on first use and stays in VRAM
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model + tokenizer into VRAM (offloads any other model first)."""
        if self._model is not None:
            return

        load_kwargs: Dict[str, Any] = dict(
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
        )

        if self._load_in_4bit:
            if HAS_BNB:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16",
                )
                print("[LLM] 4-bit quantization enabled (BitsAndBytes nf4).")
            else:
                print(
                    "[LLM] WARNING: load_in_4bit=True but bitsandbytes is not "
                    "installed. Falling back to full-precision."
                )

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs,
            )
            return tok, mdl

        tok, mdl = model_manager.get(self.model_name, _load)
        self._tokenizer = tok
        self._model = mdl

    def _offload(self) -> None:
        """Release model from VRAM. Call this manually to free memory."""
        self._model = None
        self._tokenizer = None
        model_manager.offload_current()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        self._ensure_loaded()
        return self._generate(system_prompt, user_prompt, enable_thinking)

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(
                enable_thinking
                if enable_thinking is not None
                else self.enable_thinking
            ),
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        answer = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return answer.strip()


class LLMClient:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.llm = LLM((config or CONFIG).llm)
        self.intent_classifier = IntentClassifier()

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a single formatted text prompt to the model."""
        return self.llm.chat(system_prompt, user_prompt)

    def classify_intent(self, user_input: str, context: str) -> Optional[str]:
        """Classify whether the user wants an explanation or a fix.

        .. deprecated::
            Intent classification is now handled by Laya (see
            :class:`laya_decision.LayaDecisionEngine`). This method is
            kept for backward compatibility but is no longer called by
            the planner.
        """
        return self.intent_classifier.classify(user_input, context)
