"""The model cache must distinguish *how* a model is loaded, not just its name.

A model manager keyed by name alone returns the already-loaded full model when
the user switches to 4-bit, so the quantized loader never runs. These tests pin
the composite key and the manager call it produces.
"""

from config import LLMConfig
from llm_client import LLM


def test_signature_changes_with_quantization():
    base = LLM(LLMConfig(model_name="m", quantization="none"))
    four = LLM(LLMConfig(model_name="m", quantization="4bit"))
    eight = LLM(LLMConfig(model_name="m", quantization="8bit"))

    assert base._load_signature("none") != four._load_signature("4bit")
    assert four._load_signature("4bit") != eight._load_signature("8bit")
    # The same settings produce the same key: a later request reuses the model.
    assert base._load_signature("none") == LLM(
        LLMConfig(model_name="m", quantization="none")
    )._load_signature("none")


def test_signature_includes_dtype_and_device_map():
    auto = LLM(LLMConfig(model_name="m", torch_dtype="auto", device_map="auto"))
    half = LLM(
        LLMConfig(model_name="m", torch_dtype="float16", device_map="auto")
    )
    cpu = LLM(LLMConfig(model_name="m", torch_dtype="auto", device_map="cpu"))

    assert auto._load_signature("none") != half._load_signature("none")
    assert auto._load_signature("none") != cpu._load_signature("none")


def test_quantized_auto_is_pinned_to_the_single_gpu(monkeypatch):
    """4-bit/8-bit must not take accelerate's broken CPU-offload path."""
    import llm_client

    monkeypatch.setattr(llm_client.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(llm_client.torch.cuda, "device_count", lambda: 1)

    llm = LLM(LLMConfig(
        model_name="m", quantization="4bit", device_map="auto",
    ))

    assert llm._effective_device_map("4bit") == {"": 0}
    assert llm._effective_device_map("8bit") == {"": 0}
    # Unquantized auto still uses accelerate's normal placement/offload.
    assert llm._effective_device_map("none") == "auto"
    assert '{"": 0}' in llm._load_signature("4bit")


def test_legacy_load_in_4bit_is_a_4bit_signature():
    llm = LLM(LLMConfig(model_name="m", quantization="none", load_in_4bit=True))

    quant = llm._effective_quantization()
    assert quant == "4bit"
    assert "4bit" in llm._load_signature(quant)


def test_ensure_loaded_asks_the_manager_for_the_quantized_key(monkeypatch):
    import llm_client

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeBnb:
        last = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeBnb.last = self

    monkeypatch.setattr(llm_client, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(llm_client, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(llm_client, "HAS_BNB", True)
    monkeypatch.setattr(llm_client, "BitsAndBytesConfig", FakeBnb, raising=False)

    seen = {}

    def fake_get(key, loader):
        seen["key"] = key
        seen["value"] = loader()
        return seen["value"]

    monkeypatch.setattr(llm_client.model_manager, "get", fake_get)

    llm = LLM(LLMConfig(model_name="m", quantization="4bit"))
    llm._ensure_loaded()

    assert "4bit" in seen["key"]
    assert llm._model is seen["value"][1]
    assert FakeBnb.last.kwargs["bnb_4bit_use_double_quant"] is True


def test_ensure_loaded_uses_current_dtype_keyword(monkeypatch):
    """transformers deprecates ``torch_dtype`` in favour of ``dtype``."""
    import llm_client

    seen = {}

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            seen.update(kwargs)
            return object()

    monkeypatch.setattr(llm_client, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(llm_client, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(
        llm_client.model_manager, "get", lambda key, loader: loader(),
    )

    llm = LLM(LLMConfig(
        model_name="m", torch_dtype="float16", device_map="cpu",
    ))
    llm._ensure_loaded()

    assert seen["dtype"] == "float16"
    assert "torch_dtype" not in seen


def test_classify_is_short_greedy_and_does_not_log(monkeypatch):
    llm = LLM(LLMConfig(model_name="m"))
    monkeypatch.setattr(llm, "_ensure_loaded", lambda: None)
    seen = {}

    def fake_generate(system_prompt, user_prompt, enable_thinking=None,
                      history=None, **kwargs):
        seen.update(kwargs)
        seen["enable_thinking"] = enable_thinking
        return "EXPLAIN"

    monkeypatch.setattr(llm, "_generate", fake_generate)

    assert llm.classify("router system", "router user", max_new_tokens=8) == "EXPLAIN"
    assert seen["max_new_tokens"] == 8
    assert seen["temperature"] == 0.0
    assert seen["do_sample"] is False
    assert seen["save_log"] is False
    assert seen["enable_thinking"] is False
