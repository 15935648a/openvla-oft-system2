from experiments.robot.libero.system2_agent import System2Agent
import torch

print(f"Check Device - CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}")

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Initializing System2Agent with {model_id}...")

try:
    agent = System2Agent(model_name=model_id)
    task = "pick up the red block and place it on the green plate"
    summary = "I see a red block and a green plate on the table. The robot is holding nothing."
    
    print(f"\nTesting generation...\nTask: {task}\nSummary: {summary}")
    subgoal = agent.next_subgoal(task, summary)
    print(f"\n[SUCCESS] Generated Subgoal: {subgoal}")
except Exception as e:
    print(f"\n[ERROR] Failed: {e}")
