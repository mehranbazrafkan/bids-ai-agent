from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

import agent_state
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

    def _unsupported_quantization_error(self) -> Optional[str]:
        """Explain a checkpoint this backend cannot load, before it fails.

        Some published checkpoints are ``compressed-tensors`` files with
        asymmetric weights in the ``pack-quantized`` layout. transformers
        asks for an unpacked zero-point while the file stores it packed, so
        the loader dies with a shape mismatch. Patching that load is not a
        real fix either: compressed-tensors decompresses the model to BF16
        on the first forward pass, which throws away the 4-bit memory saving
        this app needs on a small GPU. Say so up front, with the choice that
        does work: the base checkpoint with the app's own 4-bit path.
        """
        if not self.model_name or "/" not in self.model_name:
            return None
        try:
            config = AutoConfig.from_pretrained(self.model_name)
        except Exception:
            # A local path, an offline cache miss, or a model without a
            # config should be reported by the loader itself, not here.
            return None

        quant = getattr(config, "quantization_config", None)
        if quant is None:
            return None
        if hasattr(quant, "to_dict"):
            quant = quant.to_dict()
        if not isinstance(quant, dict):
            return None
        if quant.get("quant_method") != "compressed-tensors":
            return None

        groups = quant.get("config_groups") or {}
        asymmetric = any(
            isinstance(group, dict)
            and isinstance(group.get("weights"), dict)
            and group["weights"].get("symmetric") is False
            for group in groups.values()
        )
        if quant.get("format") == "pack-quantized" and asymmetric:
            return (
                f"{self.model_name} is a pre-quantized compressed-tensors "
                "checkpoint. The transformers backend cannot load this "
                "asymmetric packed format, and it would decompress to BF16 "
                "on first use, losing the 4-bit memory saving. Choose the "
                "base model (for example Qwen/Qwen3-1.7B) and set "
                "Quantization to 4-bit instead."
            )
        return None

    def _effective_device_map(self, quant: str) -> Any:
        """Place a quantized model entirely on the one available GPU.

        ``device_map="auto"`` can decide to put a few quantized blocks on the
        CPU when free VRAM is low. That path is broken in the current
        accelerate/bitsandbytes combination: accelerate calls
        ``state_dict()`` on a 4-bit module whose quantization state still has
        meta tensors, and ``Tensor.item()`` then raises. On a single-GPU
        machine, pinning the model to GPU 0 avoids the CPU-offload hooks
        entirely and gives a normal CUDA out-of-memory error if it truly does
        not fit.
        """
        if (
            quant in ("4bit", "8bit")
            and self._device_map == "auto"
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1
        ):
            return {"": 0}
        return self._device_map

    def _load_signature(self, quant: str) -> str:
        """Cache key that distinguishes two loads of the same model name.

        ``model_manager`` keeps one model per key. Keyed by name alone, a
        switch from the checkpoint dtype to 4-bit would return the already
        loaded full model and the quantized loader would never run. Every
        setting that changes the weights or their placement is part of the
        key.
        """
        device_map = self._effective_device_map(quant)
        device_map = (
            json.dumps(device_map, sort_keys=True, default=str)
            if isinstance(device_map, dict)
            else str(device_map)
        )
        return f"{self.model_name}|{quant}|{self._torch_dtype}|{device_map}"

    def _ensure_loaded(self) -> None:
        """Load model + tokenizer into VRAM (offloads any other model first)."""
        if self._model is not None:
            return

        # A client watching /predict cannot see this part, and it is the
        # slowest part of a first request. Say so, so its clock waits here
        # instead of reporting the loader as generation.
        agent_state.loading()

        unsupported = self._unsupported_quantization_error()
        if unsupported:
            raise RuntimeError(unsupported)

        quant = self._effective_quantization()
        # transformers 4.56+ renamed the ``from_pretrained`` keyword from
        # ``torch_dtype`` to ``dtype``. The config/API field keeps the old
        # name for compatibility, but only the new kwarg is passed through.
        load_kwargs: Dict[str, Any] = dict(
            dtype=self._torch_dtype,
            device_map=self._effective_device_map(quant),
        )

        # Quantization: "none" keeps the checkpoint exactly as published. The
        # legacy ``load_in_4bit`` flag still means 4-bit when nothing else
        # has chosen, so an old config.json behaves as it always did.
        if quant in ("4bit", "8bit"):
            if HAS_BNB:
                kwargs: Dict[str, Any] = dict(
                    load_in_4bit=(quant == "4bit"),
                )
                if quant == "4bit":
                    kwargs.update(
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype="float16",
                        # Quantize the quantization constants too; this saves
                        # roughly another 0.1-0.2 GB on a 1.7B model, which
                        # is enough to keep more layers on a small GPU.
                        bnb_4bit_use_double_quant=True,
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

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 8,
    ) -> str:
        """Short, deterministic generation for intent routing.

        This is intentionally separate from :meth:`chat`: it uses greedy
        decoding, disables thinking, caps the answer length and never
        records the prompt log. The classifier prompt is not the user's
        answer.
        """
        self._ensure_loaded()
        return self._generate(
            system_prompt,
            user_prompt,
            enable_thinking=False,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            save_log=False,
        )

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: Optional[bool] = None,
        history: Optional[List[Dict[str, str]]] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        save_log: bool = True,
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

        # The record of this request travels back to the client with the
        # answer (see ``prompt_log``): the system prompt exactly as the
        # planner assembled it, the user's own question, and below, what
        # the model generated and how long it took. Short classification
        # calls opt out: they are routing metadata, not the user's answer.
        started = time.perf_counter()
        if save_log:
            prompt_log.begin(system_prompt, user_prompt)

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
        effective_temperature = (
            float(self.temperature) if temperature is None else float(temperature)
        )
        sample = (
            effective_temperature > 0.0
            if do_sample is None
            else bool(do_sample)
        )
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=(
                self.max_new_tokens
                if max_new_tokens is None
                else max(1, int(max_new_tokens))
            ),
            repetition_penalty=self.repetition_penalty,
        )
        if sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=max(0.01, effective_temperature),
                top_p=float(self.top_p),
                top_k=int(self.top_k),
            )

        # From here on the model is resident and writing: the moment a
        # waiting client's clock starts. Set before the call that blocks so
        # the phase is never "loading" while tokens are being produced.
        agent_state.generating()
        outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        answer = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        # The full picture of the request -- the system prompt, the user's
        # own question, what the model answered and how it was produced
        # (checkpoint, token counts, seconds) -- is recorded in memory and
        # handed back to the client by /predict (see ``prompt_log``). It
        # replaces the entry of the previous request. A failure here never
        # costs an answer. Short classification calls opt out: they are
        # routing metadata, not the user's answer.
        if save_log:
            prompt_log.record_generation(
                answer,
                model=self.model_name,
                prompt_tokens=int(inputs.input_ids.shape[-1]),
                generated_tokens=int(generated_ids.shape[-1]),
                seconds=time.perf_counter() - started,
            )
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
