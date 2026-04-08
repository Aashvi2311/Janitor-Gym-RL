import subprocess
import os
from openenv.core.env_server import Environment
from models import JanitorAction, JanitorObservation, JanitorState

class JanitorEnv(Environment):
    def __init__(self):
        super().__init__()
        self._state = JanitorState()
        self.max_steps = 10

    def reset(self, task_id="fix_syntax") -> JanitorObservation:
        self._state.task_id = task_id
        self._state.step_count = 0
        
        # Scenario setup
        tasks = {
            "fix_syntax": "import pandas as pnd\nprint(pd.DataFrame())",
            "handle_nan": "import pandas as pd\nimport numpy as np\ndf=pd.DataFrame({'a':[1,np.nan]})\nprint(df['a'].mean())",
            "optimize_acc": "print(0.65)"
        }
        with open("train.py", "w") as f:
            f.write(tasks[task_id])
            
        return self._get_obs("Env Reset", 0, 0.0)

    @property
    def state(self) -> JanitorState:
        return self._state

    def _get_obs(self, out, code, acc):
        return JanitorObservation(
            output=out, 
            exit_code=code, 
            accuracy=acc, 
            current_step=self._state.step_count, 
            task_id=self._state.task_id
        )

    def step(self, action: JanitorAction):
        self._state.step_count += 1
        done = self._state.step_count >= self.max_steps
        reward = -0.01 

        if action.action_type == "write_file":
            with open(action.file_path, "w") as f:
                f.write(action.content)
            return self._get_obs("File Updated", 0, 0.0), 0.05, done, {}

        if action.action_type == "run_script":
            res = subprocess.run(["python", "train.py"], capture_output=True, text=True)
            acc = 0.0
            if res.returncode == 0:
                try:
                    acc = float(res.stdout.strip().split('\n')[-1])
                except:
                    acc = 1.0 if self._state.task_id != "optimize_acc" else 0.0
            
            if res.returncode == 0:
                reward = 1.0 if self._state.task_id != "optimize_acc" else max(0.0, (acc - 0.65) / 0.20)
                done = True
            else:
                reward = -0.1
            
            return self._get_obs(res.stdout + res.stderr, res.returncode, acc), reward, done, {}