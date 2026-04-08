import os
from openai import OpenAI
from env import JanitorEnv
from models import JanitorAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN: raise ValueError("HF_TOKEN is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
env = JanitorEnv()

def run_inference(task_name="fix_syntax"):
    obs = env.reset(task_id=task_name)
    print(f"[START] task={task_name} env=janitor-gym model={MODEL_NAME}")
    
    rewards = []
    done = False
    step = 0
    last_error = "null"

    while not done and step < 5:
        step += 1
        # Baseline logic: Try to run, then try to fix (Easy task fix simulation)
        action_type = "run_script"
        content = None
        if step == 2:
            action_type = "write_file"
            content = "import pandas as pd\nprint(1.0)"
        
        action = JanitorAction(action_type=action_type, file_path="train.py", content=content)
        obs, reward, done, info = env.step(action)
        
        rewards.append(f"{reward:.2f}")
        print(f"[STEP] step={step} action={action_type} reward={reward:.2f} done={str(done).lower()} error={last_error}")

    print(f"[END] success={str(reward >= 1.0).lower()} steps={step} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_inference()