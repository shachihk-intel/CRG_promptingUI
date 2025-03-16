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

# Create feedback file with headers if it doesn't exist
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=[
        "timestamp", "task", "model", "prompt", "context", 
        "response1", "response2", "response3", "response4", "response5", 
        "selected_response", "feedback_score", "feedback_text"
    ]).to_csv(FEEDBACK_FILE, index=False)

# Initialize global variables
pipeline = None
tokenizer = None

def load_model(model_choice):
    """Load the selected model and return a status message"""
    global pipeline, tokenizer
    
    try:
        # Update UI to show loading status
        status = f"Loading {model_choice}... Please wait."
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        
        # Handle different model types
        if "GPTQ" in model_choice:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_choice,
                tokenizer=tokenizer, 
                device_map="auto"
            )
        elif "tiny" in model_choice.lower():
            pipeline = transformers.pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
        else:
            # For OpenVINO models
            device = 'CPU'
            pipeline = openvino_genai.LLMPipeline(model_choice, device)
        
        return f"✅ {model_choice} loaded successfully! Ready to generate responses."
    
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def update_task_template(task_choice):
    """Update prompt and context based on selected task"""
    if task_choice == "Keyword Generation":
        prompt = "The given conversation could have multiple responses. Give out a keyword, one per response, for each possible diverse response."
        context = "Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nPossible Response Keywords: "
    
    elif task_choice == "Keyword-based Response Generation":
        prompt = "Generate a response as the user, using the given keyword.\nKeyword: sick"
        context = "Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nUser: "
    
    elif task_choice == "Direct Response Generation":
        prompt = "Generate a response for the given conversation"
        context = "Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nUser: "
    
    return prompt, context

def generate_responses(task, context, prompt, model_choice):
    """Generate multiple responses using the loaded model"""
    global pipeline
    
    if pipeline is None:
        return ["Please load a model first"] * 5
    
    try:
        # Configure generation parameters
        config = openvino_genai.GenerationConfig()
        config.num_beams = 5
        config.num_return_sequences = 5
        config.max_new_tokens = 50  # Increased token count for more substantive responses
        
        # Generate responses
        results = pipeline.generate(prompt + context, config)
        
        # Extract generated texts
        responses = results.texts if hasattr(results, 'texts') else []
        
        # Fill with empty strings if we don't have enough responses
        while len(responses) < 5:
            responses.append("")
        
        # Log generation details
        log_generation(task, model_choice, prompt, context, responses)
        
        return responses[:5]
    
    except Exception as e:
        print(f"Error generating responses: {str(e)}")
        return [f"Error: {str(e)}"] + [""] * 4

def log_generation(task, model, prompt, context, responses):
    """Log generation details to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_data = {
        "timestamp": timestamp,
        "task": task,
        "model": model,
        "prompt": prompt,
        "context": context
    }
    
    # Add responses
    for i, resp in enumerate(responses[:5], 1):
        log_data[f"response{i}"] = resp
    
    # Create or append to log file
    df = pd.DataFrame([log_data])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def save_feedback(task, model, prompt, context, response1, response2, response3, response4, response5, 
               like_all, cb1, cb2, cb3, cb4, cb5, rating, feedback_text):
    """Save user feedback to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare selected responses string
    selected_responses = []
    if like_all:
        selected_responses.append("All Responses")
    else:
        if cb1: selected_responses.append("Response 1")
        if cb2: selected_responses.append("Response 2")
        if cb3: selected_responses.append("Response 3")
        if cb4: selected_responses.append("Response 4")
        if cb5: selected_responses.append("Response 5")
    
    selected = ", ".join(selected_responses) if selected_responses else "None selected"
    
    # Prepare data for saving
    feedback_data = {
        "timestamp": timestamp,
        "task": task,
        "model": model,
        "prompt": prompt,
        "context": context,
        "response1": response1,
        "response2": response2,
        "response3": response3,
        "response4": response4,
        "response5": response5,
        "selected_response": selected,
        "feedback_score": rating,
        "feedback_text": feedback_text
    }
    
    # Save to CSV
    df = pd.DataFrame([feedback_data])
    df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    
    return "✅ Feedback saved successfully! Thank you for your input."

# Define a function to toggle checkbox interactivity
def toggle_response_checkboxes(like_all):
    """Disable/enable response checkboxes based on 'like all' checkbox"""
    return [gr.update(interactive=not like_all) for _ in range(5)]

def load_feedback_data():
    """Load saved feedback data for display"""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame()

# Build the UI
with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(f"{DESCRIPTION}")
    
    with gr.Tabs():
        with gr.TabItem("Generate Responses"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    gr.Markdown("### Model Selection")
                    model_choice = gr.Radio(
                        choices=MODELS, 
                        label="Select Model",
                        info="Choose the language model to use",
                        container=False
                    )
                    model_status = gr.Textbox(
                        label="Model Status", 
                        value="No model loaded", 
                        interactive=False
                    )
                    load_model_btn = gr.Button("Load Selected Model")
                    
                    gr.Markdown("### Task Configuration")
                    task_choice = gr.Radio(
                        choices=TASKS, 
                        label="Task Type", 
                        value=TASKS[0],
                        info="Choose the type of generation task"
                    )
                    prompt = gr.Textbox(
                        label="Prompt", 
                        info="Instructions for the model",
                        lines=3
                    )
                    context = gr.Textbox(
                        label="Conversation Context", 
                        info="The conversation history or context for generation",
                        lines=5
                    )
                    
                    generate_btn = gr.Button("Generate Responses")
                
                with gr.Column(scale=2):
                    # Results section
                    gr.Markdown("### Generated Responses")
                    response_boxes = []
                    for i in range(5):
                        response_box = gr.Textbox(
                            label=f"Response {i+1}", 
                            lines=3, 
                            interactive=False
                        )
                        response_boxes.append(response_box)
                    
                    # Feedback section
                    gr.Markdown("### Provide Feedback")
                    gr.Markdown("Select your favorite response(s):")
                    with gr.Row():
                        like_all_responses = gr.Checkbox(label="I like all responses")
                    
                    with gr.Row():
                        response_checkboxes = []
                        for i in range(5):
                            cb = gr.Checkbox(label=f"Response {i+1}")
                            response_checkboxes.append(cb)
                    feedback_rating = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        step=1, 
                        label="Rate Quality (1-5)", 
                        value=3
                    )
                    feedback_text = gr.Textbox(
                        label="Additional Feedback (Optional)", 
                        placeholder="What did you like or dislike about the responses?",
                        lines=2
                    )
                    save_btn = gr.Button("Save Feedback")
                    feedback_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.TabItem("View Previous Feedback"):
            gr.Markdown("### Previously Collected Feedback")
            refresh_btn = gr.Button("Refresh Data")
            feedback_data = gr.Dataframe(label="Feedback Data")
    
    # Set initial default values for prompt and context
    initial_prompt, initial_context = update_task_template(TASKS[0])
    prompt.value = initial_prompt
    context.value = initial_context
    
    # Set up event handlers
    load_model_btn.click(
        fn=load_model, 
        inputs=[model_choice], 
        outputs=[model_status]
    )
    
    task_choice.change(
        fn=update_task_template, 
        inputs=[task_choice], 
        outputs=[prompt, context]
    )
    
    generate_btn.click(
        fn=generate_responses, 
        inputs=[task_choice, context, prompt, model_choice], 
        outputs=response_boxes
    )
    
    save_btn.click(
        fn=save_feedback,
        inputs=[
            task_choice, model_choice, prompt, context, 
            response_boxes[0], response_boxes[1], response_boxes[2], response_boxes[3], response_boxes[4],
            like_all_responses, 
            response_checkboxes[0], response_checkboxes[1], response_checkboxes[2], response_checkboxes[3], response_checkboxes[4],
            feedback_rating, feedback_text
        ],
        outputs=[feedback_status]
    )
    
    # Add interaction between "like all" checkbox and the individual response checkboxes
    def toggle_response_checkboxes(like_all):
        return [gr.update(interactive=not like_all) for _ in range(5)]
    
    like_all_responses.change(
        fn=toggle_response_checkboxes,
        inputs=[like_all_responses],
        outputs=response_checkboxes
    )
    
    refresh_btn.click(
        fn=load_feedback_data,
        inputs=[],
        outputs=[feedback_data]
    )

# Set default values on page load
def set_defaults():
    initial_prompt, initial_context = update_task_template(TASKS[0])
    return initial_prompt, initial_context

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        share=False,
        debug=False,
        server_name="0.0.0.0", 
        server_port=8051
    )