from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self):
        LLM_name = "google/gemma-3-1b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_name,
            # torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def chat(self, prompt: str, tools=None):
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=200,
            temperature=0.2
        )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        return answer

class LLMClient:
    def __init__(self):
        self.model = LLM()
        pass

    def generate_response(self, prompt: str) -> str:
        """Sends a single formatted text prompt to the model."""
        if not self.model:
            # Placeholder response if model_instance is not injected
            return f"[LLM Response Simulated for Prompt length {len(prompt)}]"

        return self.model.chat(prompt)

    def tool_chat(self, prompt: str, tools_schema: List[Dict[str, Any]]) -> str:
        """Sends a prompt along with available tool definitions for structured decision-making."""
        if not self.model:
            # Placeholder decision simulation for the Fix workflow
            return "TOOL_CALL: repair_metadata"

        # return self.model.chat(prompt, tools=tools_schema)
        return self.model.chat(prompt)