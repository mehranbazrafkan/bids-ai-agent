from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

import prompt_log
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
        self._quantization = getattr(cfg, "quantization", "none")

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy loading — model is loaded on first use and stays in VRAM
    # ------------------------------------------------------------------

    def _effective_quantization(self) -> str:
        """Normalise ``quantization``/legacy ``load_in_4bit`` to one choice."""
        quant = str(self._quantization or "none").strip().lower()
        if quant not in ("none", "4bit", "8bit"):
            print(
                f"[LLM] WARNING: unknown quantization {quant!r}; "
                "using the unquantized checkpoint dtype."
            )
            quant = "none"
        if quant == "none" and self._load_in_4bit:
            quant = "4bit"
        return quant

    def _load_signature(self, quant: str) -> str:
        """Cache key that distinguishes two loads of the same model name.

        ``model_manager`` keeps one model per key. Keyed by name alone, a
        switch from the checkpoint dtype to 4-bit would return the already
        loaded full model and the quantized loader would never run. Every
        setting that changes the weights or their placement is part of the
        key.
        """
        device_map = (
            json.dumps(self._device_map, sort_keys=True, default=str)
            if isinstance(self._device_map, dict)
            else str(self._device_map)
        )
        return f"{self.model_name}|{quant}|{self._torch_dtype}|{device_map}"

    def _ensure_loaded(self) -> None:
        """Load model + tokenizer into VRAM (offloads any other model first)."""
        if self._model is not None:
            return

        quant = self._effective_quantization()
        load_kwargs: Dict[str, Any] = dict(
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
        )

        # Quantization: "none" keeps the checkpoint exactly as published. The
        # legacy ``load_in_4bit`` flag still means 4-bit when nothing else
        # has chosen, so an old config.json behaves as it always did.
        if quant in ("4bit", "8bit"):
            if HAS_BNB:
                kwargs: Dict[str, Any] = dict(load_in_4bit=(quant == "4bit"))
                if quant == "4bit":
                    kwargs.update(
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype="float16",
                    )
                load_kwargs["quantization_config"] = BitsAndBytesConfig(**kwargs)
                print(f"[LLM] {quant} quantization enabled (BitsAndBytes).")
            else:
                print(
                    f"[LLM] WARNING: quantization={quant} but bitsandbytes is "
                    "not installed. Falling back to the unquantized checkpoint dtype."
                )

        def _load():
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs,
            )
            return tok, mdl

        signature = self._load_signature(quant)
        tok, mdl = model_manager.get(signature, _load)
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
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        self._ensure_loaded()
        return self._generate(system_prompt, user_prompt, enable_thinking, history)

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: Optional[bool] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        # The conversation travels as real chat turns, in order, between
        # the system prompt and the live question — the shape the chat
        # template itself expects, which is what makes a second question
        # read as a second question rather than a fresh start. It used to
        # be *only* pasted into the system prompt as a "[Conversation
        # History]" block, which a small model reads as scenery: asked
        # the same finding again it would answer as if nothing had been
        # said — "give me the error" — with the error in ``context`` the
        # whole time.
        #
        # The planner now renders the same turns into the system prompt
        # as well (see ``Planner.build_system_prompt``), so both halves
        # carry the conversation. This loop stays the authority on the
        # chat shape: whatever the system prompt says, the turns below
        # are the ones the template is handed.
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for turn in history or []:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip().lower()
            content = str(turn.get("content") or "")
            # Only the two roles a conversation is made of, and only
            # something to say: a template handed "system" or an empty
            # string here produces a turn the model cannot answer to.
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        if getattr(self._tokenizer, "chat_template", None):
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
        else:
            # No chat template means a base checkpoint: pre-trained,
            # never instruction-tuned (google/gemma-3-270m rather than
            # its -it sibling, say). There is no honest way to turn a
            # question into input for one — prompting it as plain text
            # only yields fluent nonsense — so say so instead, in words
            # that name the fix. /predict passes this through as JSON and
            # the desktop app shows it, rather than an HTML error page.
            raise RuntimeError(
                f"{self.model_name} has no chat template, so it cannot "
                "answer questions: it is a base (pre-trained) checkpoint, "
                "not an instruction-tuned one. Pick an -it or -instruct "
                "model in Settings -> AI Agent."
            )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
        ).to(self._model.device)

        # temperature / top_p / top_k are ignored unless generation
        # samples — transformers reports them as "not valid ... may be
        # ignored" and decodes greedily. Left alone, the "answer variety"
        # control in Settings would be a placebo, so sampling is asked for
        # explicitly. Temperature <= 0 is the one choice that should still
        # mean "always the same, most likely answer".
        sample = float(self.temperature) > 0.0
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
        )
        if sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                top_k=int(self.top_k),
            )

        outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        answer = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        # TEMPORARY (prompt inspection aid): the full picture of the
        # latest request -- the system prompt, the user's own question
        # and what the model answered -- replacing the previous entry.
        # Remove this call and the ``prompt_log`` module once testing is
        # done; a failure here never costs an answer.
        prompt_log.save(system_prompt, user_prompt=user_prompt, output=answer)
        return answer


class LLMClient:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.llm = LLM((config or CONFIG).llm)
        self.intent_classifier = IntentClassifier()

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send one prompt to the model, with the conversation so far.

        ``history`` is passed through as chat turns rather than folded
        into ``system_prompt`` — see :meth:`LLM._generate`.
        """
        return self.llm.chat(system_prompt, user_prompt, history=history)

    def classify_intent(self, user_input: str, context: str) -> Optional[str]:
        """Classify whether the user wants an explanation or a fix.

        .. deprecated::
            Intent classification is now handled by Laya (see
            :class:`laya_decision.LayaDecisionEngine`). This method is
            kept for backward compatibility but is no longer called by
            the planner.
        """
        return self.intent_classifier.classify(user_input, context)
