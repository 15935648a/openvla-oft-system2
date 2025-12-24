import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class System2Agent:
    def __init__(self, model_name="nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", api_key=None):
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
                    except Exception as e_slow:
                        logger.warning(f"Failed to load slow tokenizer: {e_slow}, trying force_download=True...")
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

    def summarize_observation(self, obs):
        """
        Summarizes the low-level robot state into text for the LLM.
        """
        # For now, let's at least mention we are in a simulation. 
        # In a full system, this would use a VLM or object detector.
        return "The robot is in the starting position, looking at the tabletop objects."

    def next_subgoal(self, task_description, obs_summary):
        """
        Decides the next subgoal using a chat format for better instruction following.
        """
        messages = [
            {"role": "system", "content": "You are a robotic controller. Given a task and a state, output ONLY the next immediate physical subgoal as a short, simple sentence."},
            {"role": "user", "content": f"Task: {task_description}\nState: {obs_summary}\n\nWhat is the next immediate step? Output it starting with 'Action:'"}
        ]

        if self.is_local and self.model:
            # Use chat template if available, else fallback to manual format
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                prompt = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant: Action:"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.4, # Lower temperature for more consistency
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's new response
            if "Assistant:" in full_text:
                response = full_text.split("Assistant:")[-1].strip()
            elif "Action:" in full_text:
                response = full_text.split("Action:")[-1].strip()
            else:
                response = full_text.replace(prompt, "").strip()

            # Final Cleanup: Get the very first line of the actual response
            response = response.split('\n')[0].replace("Action:", "").strip()
            
            # Cleaning: remove quotes and common rambling
            response = response.strip('"').strip("'").strip(".")
            
            # Sanity Check
            if len(response) < 5 or any(c in response for c in ["_", "[", "{", ">"]):
                return task_description
            
            return response

        return task_description
