# Janitor-Gym: MLOps Agentic Environment

**Janitor-Gym** is a specialized OpenEnv-compliant environment designed to train and evaluate AI agents on real-world MLOps tasks. Instead of toy problems, agents are tasked with debugging code, handling data pipeline failures, and optimizing model performance on real datasets (UCI Student Dropout & Olist E-Commerce).

## 🚀 Overview
Modern LLMs are great at writing code but often fail at iterative debugging and complex environment feedback loops. Janitor-Gym provides a **deterministic, tool-use environment** where an agent must act as a "Junior ML Engineer" to fix a broken production pipeline.

## 🛠️ Environment Specifications
- **Observation Space:** Structured logs, stack traces, and data schemas.
- **Action Space:** `read_file`, `write_file`, and `run_script`.
- **Reward Function:** Incremental progress rewards (+0.05 for file updates) and high-stakes terminal rewards (1.0 for passing task-specific graders). Penalties applied for syntax errors (-0.1) and time-to-completion (-0.01 per step).

## 📊 Tasks & Difficulty
| Task ID | Level | Objective | Grading Criteria |
| :--- | :--- | :--- | :--- |
| `fix_syntax` | **Easy** | Fix broken imports/syntax in `train.py`. | Binary: 1.0 if script runs. |
| `handle_nan` | **Medium** | Implement imputation for missing values. | Functional: 1.0 if training completes. |
| `optimize_acc` | **Hard** | Hyper-parameter tuning / Feature Eng. | Scaled: Score based on accuracy vs baseline. |

## 💻 Technical Setup
### Prerequisites
- Python 3.10+
- `openenv-core`
- `openai`

### Local Installation
```bash
pip install -r requirements.txt
