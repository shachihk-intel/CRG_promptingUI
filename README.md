# CRG_promptingUI
This is the code for the prompting experiments in CRG for ACAT

## Install the requirements
Step1: Create and activate a virtual environment 
```bash
python -m venv prompting_env
source prompting_env/bin/activate  # On Windows: prompting_env\Scripts\activate
```

Step2: Install the requirements 
```bash
pip install -r requirements.txt
```

## Convert your models to openvino format
```bash
# Convert Llama models to OpenVINO format. The UI currently provides three choices (as below). But any additional model can be easily added. 
optimum-cli export openvino --trust-remote-code --model meta-llama/Llama-3.2-3B-Instruct Llama-3.2-3B-Instruct
optimum-cli export openvino --trust-remote-code --model meta-llama/Llama-2-7b-chat-hf Llama-2-7b-chat-hf
optimum-cli export openvino --trust-remote-code --model meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct
```

## Running the UI
To run the UI and start generating responses:

```bash
python ui.py
```

This will start the Gradio interface at http://0.0.0.0:8051

## UI Features
The UI provides the following features:

1. **Model Selection**: Choose from Llama-2-7b, Llama-3-8B, or Llama-3.2-3B models. You can add any new models by adding to the "MODELS" variable in ui.py and using optimum-cli to export the model. 
2. **Task Types**:
   - Direct Response Generation
   - Keyword Generation
   - Keyword-based Response Generation
3. **Response Generation**: Generate up to 5 diverse responses
4. **Feedback Collection**: Collect user feedback including:
   - Best response selection
   - Option to like all responses
   - Quality rating (1-5)
   - Text feedback

## Code Structure
The main UI code is structured as follows:

```python
import gradio as gr
import transformers
import torch
import openvino_genai
import os
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer

# Constants
TITLE = "CRG Prompt Engineering"
DESCRIPTION = "Try different models and prompts to generate responses, then provide feedback"
MODELS = ["Llama-2-7b-chat-hf", "Meta-Llama-3-8B-Instruct", "Llama-3.2-3B-Instruct"]
TASKS = ["Direct Response Generation", "Keyword Generation", "Keyword-based Response Generation"]
FEEDBACK_FILE = "user_feedback.csv"
LOG_FILE = "generation_log.csv"

# Main UI components and functions
# ...

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        share=False,
        debug=False,
        server_name="0.0.0.0", 
        server_port=8051
    )
```

For a complete implementation, see the `ui_improved.py` file.

## Data Collection
The UI saves all feedback to a CSV file (`user_feedback.csv`) with the following structure:
- timestamp
- task type
- model used
- prompt
- context
- all responses
- selected response (or "All Responses")
- feedback score
- feedback text

Generation logs are saved to `generation_log.csv` for analysis.
