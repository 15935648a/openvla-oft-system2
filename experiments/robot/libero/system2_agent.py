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
        import numpy as np
        state_parts = []

        # 1. Gripper State
        # In LIBERO/Robosuite: gripper qpos usually 2 dims.
        # OpenVLA normalization: -1 (open) to 1 (closed) or vice versa depending on dataset.
        # But raw simulation values: 0 (closed) to ~0.08 (open).
        if "robot0_gripper_qpos" in obs:
            gripper_qpos = obs["robot0_gripper_qpos"]
            # Heuristic: if values are very small, it's closed/holding.
            # If values are larger (close to max), it's open.
            if np.mean(gripper_qpos) < 0.03: 
                 state_parts.append("The robot gripper is CLOSED (likely holding something).")
            else:
                 state_parts.append("The robot gripper is OPEN (holding nothing).")
        
        # 2. Ground Truth Object Info (if available in obs)
        # Scan for keys that look like object positions
        objects_found = []
        for key in obs.keys():
            if "pos" in key and "robot" not in key and "eef" not in key and "joint" not in key and "gripper" not in key:
                obj_name = key.replace("_pos", "").replace("_", " ")
                objects_found.append(obj_name)
        
        if objects_found:
            state_parts.append(f"I see the following objects: {', '.join(objects_found)}.")
        else:
            state_parts.append("The robot is facing the workspace.")

        return " ".join(state_parts)

    def next_subgoal(self, task_description, obs_summary):
        """
        Decides the next subgoal using few-shot prompting to force decomposition.
        """
        # Improved few-shot examples with kitchen/manipulation context
        examples = """
Task: put the red block on the green plate
State: The robot gripper is OPEN (holding nothing). I see a red block and a green plate.
Action: Pick up the red block

Task: put both the soup and the sauce in the basket
State: The robot gripper is OPEN (holding nothing). I see a soup can, a sauce bottle, and a basket.
Action: Pick up the soup can

Task: put both the soup and the sauce in the basket
State: The robot gripper is CLOSED (likely holding something).
Action: Place the soup can in the basket

Task: turn on the stove
State: The robot gripper is OPEN (holding nothing). I see a stove.
Action: Turn on the stove
"""
        
        system_prompt = "You are a robot logic unit. Your job is to break complex tasks into the FIRST single step based on the current state. Never repeat the full task."
        user_prompt = f"{examples}\nTask: {task_description}\nState: {obs_summary}\nAction:"

        if self.is_local and self.model:
            # Construct prompt
            prompt = f"{system_prompt}\n\n{user_prompt}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60, 
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Robust Parsing
            if "Action:" in full_text:
                # Get text after the LAST "Action:"
                response = full_text.split("Action:")[-1].strip()
            else:
                # Fallback: try to see if the model just outputted the action directly
                # Remove the prompt from the response
                response = full_text.replace(prompt, "").strip()

            # Clean up first line only
            response = response.split('\n')[0].strip()
            response = response.strip('"').strip("'").strip(".")
            
            # Validation
            if not response or len(response) < 3 or "Task:" in response:
                logger.warning(f"System 2 generated invalid subgoal: '{response}'. Falling back to full task.")
                return task_description
            
            return response

        return task_description
