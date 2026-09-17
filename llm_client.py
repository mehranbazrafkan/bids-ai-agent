from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM:
    def __init__(self):
        # model_name = "google/gemma-3-1b-it"
        # model_name = "google/gemma-3-270m"
        # model_name = "google/gemma-3-4b-it" # Is not downloaded (8GB)
        
        # model_name = "Qwen/Qwen3-4B" # Is not downloaded (8GB)
        model_name = "Qwen/Qwen3-1.7B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def chat(
        self,
        system_prompt: str,
        prompt: str
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # For simple models like Gemma-3-270M
        # User:
        # text = f"""System:
        # {system_prompt}

        # {prompt}

        # Assistant:
        # """

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # For Models that support "thinking" mode, this can be set to True to enable it.
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=1000,
            temperature=0.6,
            top_p=0.95,
            top_k=20
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return answer.strip()

class LLMClient:
    def __init__(self):
        self.model = LLM()
        pass

    def generate_response(self, system_prompt: str, prompt: str) -> str:
        """Sends a single formatted text prompt to the model."""
        if not self.model:
            # Placeholder response if model_instance is not injected
            return f"[LLM Response Simulated for Prompt length {len(prompt)}]"

        return self.model.chat(system_prompt, prompt)

    def tool_chat(self, system_prompt: str, prompt: str, tools_schema: List[Dict[str, Any]]) -> str:
        """Sends a prompt along with available tool definitions for structured decision-making."""
        if not self.model:
            # Placeholder decision simulation for the Fix workflow
            return "TOOL_CALL: repair_metadata"

        # return self.model.chat(prompt, tools=tools_schema)
        return self.model.chat(system_prompt, prompt)