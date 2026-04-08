from openenv.core.env_server import Action, Observation, State
from typing import Literal, Optional

class JanitorAction(Action):
    action_type: Literal["read_file", "write_file", "run_script"]
    file_path: str
    content: Optional[str] = None

class JanitorObservation(Observation):
    output: str
    exit_code: int
    accuracy: float
    current_step: int
    task_id: str

class JanitorState(State):
    step_count: int = 0
    task_id: str = "fix_syntax"