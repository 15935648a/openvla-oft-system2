import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class System2Agent:
    def __init__(self, model_name="gpt-4o", api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.is_local = False
        self.model = None
        self.tokenizer = None

        # Check if it is likely a local Hugging Face model (not GPT/Claude)
        if "gpt" not in model_name.lower() and "claude" not in model_name.lower():
            self.is_local = True
            logger.info(f"Loading local System 2 model: {model_name}")
            try:
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                logger.info(f"Using device: {device}")

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                except Exception:
                    logger.warning("Failed to load fast tokenizer, trying slow tokenizer...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
                    except Exception:
                        logger.warning("Failed to load slow tokenizer, trying force_download=True...")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, force_download=True)

                # Load model without device_map="auto" to avoid CUDA requirement on MPS
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=None,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device)
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load local model {model_name}: {e}")
                logger.error(f"Could not load System 2 model. Running in pass-through mode (System 1 only). Error: {e}")
                self.model = None
                self.tokenizer = None
        else:
            logger.info(f"Initialized System2Agent with API model {model_name}")

    def next_subgoal(self, task_description, obs_summary):
        """
        Decides the next subgoal based on the high-level task and current observation summary.
        """
        prompt = f"""You are a robotic planning assistant.
High-level Task: {task_description}
Current Observation: {obs_summary}

What is the next immediate short-term subgoal to achieve the high-level task?
Reply with ONLY the subgoal text.
Subgoal:"""

        if self.is_local and self.model:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Simple logic to extract response if model echoes prompt
            if prompt in generated_text:
                response = generated_text.split("Subgoal:")[-1].strip().split('\n')[0]
            else:
                response = generated_text.strip().split('\n')[0]
            return response

        # Fallback or API logic
        return task_description

    def summarize_observation(self, obs):
        # Placeholder for VLM summary logic
        return "Standard robot state"
