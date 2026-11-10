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
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
