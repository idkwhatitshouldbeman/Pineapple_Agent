import streamlit as st
import requests
import json
import base64
import time
import re
import io
import sys
import contextlib
from PIL import Image
import uuid
from datetime import datetime

#starting code: python -m streamlit run C:\Users\arvin\Downloads\PineFile.py

# Configuration dictionary for available models
CONFIG = {
    "planning_agents": [
        {"name": "Claude 3.5 Sonnet", "id": "anthropic/claude-3.5-sonnet"},
        {"name": "Grok 3", "id": "xai/grok-3"},
        {"name": "Llama 3.1 70B", "id": "meta-llama/llama-3.1-70b-instruct"},
        {"name": "Mistral Large 2", "id": "mistralai/mixtral-8x22b-instruct"},
        {"name": "Grok 3 Mini", "id": "xai/grok-3-mini"}
    ],
    "coding_agents": [
        {"name": "Claude 3.5 Sonnet", "id": "anthropic/claude-3.5-sonnet"},
        {"name": "Llama 3.1 70B", "id": "meta-llama/llama-3.1-70b-instruct"},
        {"name": "Grok 3", "id": "xai/grok-3"},
        {"name": "Mistral Large 2", "id": "mistralai/mixtral-8x22b-instruct"},
        {"name": "Grok 3 Mini", "id": "xai/grok-3-mini"}
    ],
    "checking_agents": [
        {"name": "Claude 3.5 Sonnet", "id": "anthropic/claude-3.5-sonnet"},
        {"name": "Grok 3", "id": "xai/grok-3"},
        {"name": "Llama 3.1 70B", "id": "meta-llama/llama-3.1-70b-instruct"},
        {"name": "Mistral Large 2", "id": "mistralai/mixtral-8x22b-instruct"},
        {"name": "Grok 3 Mini", "id": "xai/grok-3-mini"}
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}
# Set page configuration
# Page config is already set at the beginning of the script

# Define CSS for dark theme and clean UI
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: pink;

    }
    .stTextInput, .stTextArea {
        background-color: #333333;
        color: pink;

        border-radius: 15px;
        border: 1px solid #FFFFFF;
    }
    .stButton>button {
        background-color: #333333;
        color: pink;

        border-radius: 15px;
        border: 1px solid #FFFFFF;
        padding: 12px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4D4D4D;
    }
    .task-header {
        color: pink;

        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
    }
    .task-content {
        background-color: #333333;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .completed-task {
        color: pink;

        text-decoration: line-through;
    }
    .current-task {
        color: pink;

        font-weight: bold;
    }
    .future-task {
        color: #CCCCCC;
    }
    .code-export {
        background-color: #333333;
        border-radius: 15px;
        padding: 15px;
        font-family: monospace;
        white-space: pre-wrap;
        color: pink;
    }
    .model-selection {
        background-color: #333333;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .avatar-container {
        background-color: #333333;
        border-radius: 50%;
        padding: 10px;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
    }
    .chat-message {
        display: flex;
        margin-bottom: 15px;
    }
    .message-content {
        background-color: #333333;
        border-radius: 15px;
        padding: 15px;
        max-width: 80%;
        color: pink;
    }
    .user-message {
        justify-content: flex-end;
    }
    .user-message .message-content {
        background-color: #4D4D4D;
    }
    .bot-message {
        justify-content: flex-start;
    }
    .progress-container {
        background-color: #333333;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .tab-content {
        padding: 15px;
        background-color: #333333;
        border-radius: 15px;
    }
    .main-title {
        text-align: center;
        color: pink;
        font-size: 36px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: pink;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .uploaded-image {
        max-width: 100%;
        margin: 10px 0;
        border-radius: 15px;
    }
    .error-message {
        color: pink;
        background-color: #4D4D4D;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .success-message {
        color: pink;
        background-color: #333333;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .stWarning {
        font-size: 24px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        color: #FFFFFF !important;
        background-color: #4D4D4D !important;
        border-radius: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'current_task_index' not in st.session_state:
    st.session_state.current_task_index = 0
if 'code_output' not in st.session_state:
    st.session_state.code_output = ""
if 'execution_result' not in st.session_state:
    st.session_state.execution_result = ""
if 'api_key' not in st.session_state:
    st.session_state.api_key = "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"
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Define the free OpenRouter models
FREE_MODELS = {
    "Planning": [
        "anthropic/claude-3.5-sonnet",
        "xai/grok-3",
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mixtral-8x22b-instruct",
        "xai/grok-3-mini"
    ],
    "Coding": [
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-70b-instruct",
        "xai/grok-3",
        "mistralai/mixtral-8x22b-instruct",
        "xai/grok-3-mini"
    ],
    "Checking": [
        "anthropic/claude-3.5-sonnet",
        "xai/grok-3",
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mixtral-8x22b-instruct",
        "xai/grok-3-mini"
    ]
}

# Initialize model selections in session state
for role in FREE_MODELS:
    if f'{role.lower()}_model' not in st.session_state:
        st.session_state[f'{role.lower()}_model'] = FREE_MODELS[role][0]
    if f'{role.lower()}_index' not in st.session_state:
        st.session_state[f'{role.lower()}_index'] = 0

# Function to make API call to OpenRouter
def query_openrouter(prompt, model, system_prompt="", images=None):
    headers = {
        "Authorization": f"Bearer {st.session_state.api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    
    # Handle images if provided
    if images:
        content = []
        # Add text part
        content.append({"type": "text", "text": prompt})
        
        # Add image parts
        for img_data in images:
            if isinstance(img_data, str) and img_data.startswith("data:image"):
                # Already base64 encoded
                image_data = img_data
            else:
                # Need to encode
                buffer = io.BytesIO()
                img_data.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                image_data = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
        
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            # Try fallback model on error
            return fallback_query(prompt, model, system_prompt, images)
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
        # Try fallback model on exception
        return fallback_query(prompt, model, system_prompt, images)

# Function to try fallback models
def fallback_query(prompt, current_model, system_prompt="", images=None):
    # Determine role from the current model
    role = None
    model_index = 0
    
    for r in FREE_MODELS:
        if current_model in FREE_MODELS[r]:
            role = r
            model_index = FREE_MODELS[r].index(current_model)
            break
    
    if not role:
        return "Error: Could not find a suitable fallback model."
    
    # Try remaining models in the role's list
    for i in range(1, len(FREE_MODELS[role])):
        next_index = (model_index + i) % len(FREE_MODELS[role])
        fallback_model = FREE_MODELS[role][next_index]
        
        if fallback_model != current_model:
            st.warning(f"Trying fallback model: {fallback_model}")
            
            headers = {
                "Authorization": f"Bearer {st.session_state.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            
            # Handle images if provided
            if images:
                content = []
                content.append({"type": "text", "text": prompt})
                for img_data in images:
                    if isinstance(img_data, str) and img_data.startswith("data:image"):
                        image_data = img_data
                    else:
                        buffer = io.BytesIO()
                        img_data.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        image_data = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_data}
                    })
                
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": fallback_model,
                "messages": messages
            }
            
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                if response.status_code == 200:
                    # Update the current model selection
                    st.session_state[f'{role.lower()}_model'] = fallback_model
                    st.session_state[f'{role.lower()}_index'] = next_index
                    return response.json()["choices"][0]["message"]["content"]
            except:
                continue
    
    return "Error: All models failed. Please try again later or with a different prompt."

# Function to extract code from text
def extract_code(text):
    # Look for markdown code blocks
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    
    if matches:
        return "\n\n".join(matches)
    
    # If no markdown blocks, try to extract just code-like content
    pattern = r"def .*:|class .*:|import .*|from .* import"
    if re.search(pattern, text):
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if re.search(pattern, line):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        return "\n".join(code_lines)
    
    return text

# Function to safely execute code
def execute_code(code):
    # Create string buffer
    output_buffer = io.StringIO()
    # Store original stdout
    old_stdout = sys.stdout
    sys.stdout = output_buffer
    
    result = ""
    error = None
    
    try:
        # Create local environment
        local_vars = {}
        
        # Execute with timeout
        exec(code, globals(), local_vars)
        
        result = output_buffer.getvalue()
    except Exception as e:
        error = str(e)
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    return result, error

# Function to update task list
def update_tasks(planning_result):
    # Try to extract tasks from the response
    task_pattern = r"(?:Step|Task)\s*\d+:?\s*(.*?)(?:\n|$)"
    extracted_tasks = re.findall(task_pattern, planning_result)
    
    if not extracted_tasks:
        # Try a more general pattern
        task_pattern = r"\d+\.\s*(.*?)(?:\n|$)"
        extracted_tasks = re.findall(task_pattern, planning_result)
    
    if extracted_tasks:
        st.session_state.tasks = extracted_tasks
        st.session_state.current_task_index = 0
    else:
        # If no tasks are found, create general tasks
        st.session_state.tasks = [
            "Analyze requirements",
            "Generate code solution",
            "Test the solution",
            "Refine and optimize"
        ]
        st.session_state.current_task_index = 0

# Planning function
def plan_solution(user_prompt, images=None):
    system_prompt = """You are an expert AI planning agent. Your role is to analyze user requirements and break them down into clear, actionable steps. 
    For coding tasks:
    1. Analyze what the user wants to build
    2. Research any necessary APIs, libraries, or tools
    3. Break down the task into clear steps
    4. Provide specific guidance on implementation approach
    5. Identify potential challenges and solutions
    
    Be thorough and detailed. Your plan will be used by a coding agent to implement the solution."""
    
    planning_model = st.session_state.planning_model
    
    with st.spinner("Planning solution..."):
        planning_result = query_openrouter(user_prompt, planning_model, system_prompt, images)
        
    # Update the task list based on the planning result
    update_tasks(planning_result)
    
    return planning_result

# Coding function
def generate_code(user_prompt, planning_result, images=None):
    system_prompt = """You are an expert coding AI. Your task is to implement high-quality, working code based on a user's requirements and a planning outline.
    Follow these guidelines:
    1. Write clean, well-documented code
    2. Implement all required features
    3. Handle errors appropriately
    4. Focus on creating functional solutions
    5. Comment your code to explain complex sections
    
    Provide the complete implementation. Format your code with proper markdown (```language) code blocks."""
    
    coding_model = st.session_state.coding_model
    
    combined_prompt = f"""User Request: {user_prompt}
    
    Planning Analysis:
    {planning_result}
    
    Based on this information, implement the full solution with well-structured, working code. Include appropriate error handling and documentation."""
    
    with st.spinner("Generating code..."):
        coding_result = query_openrouter(combined_prompt, coding_model, system_prompt, images)
    
    # Extract actual code from the response
    code = extract_code(coding_result)
    st.session_state.code_output = code
    
    # Move to the next task
    if st.session_state.current_task_index < len(st.session_state.tasks) - 1:
        st.session_state.current_task_index += 1
    
    return coding_result, code

# Code checking function
def check_code(user_prompt, planning_result, code, images=None):
    system_prompt = """You are an expert code review AI. Your job is to review code for correctness, efficiency, and adherence to requirements.
    Follow these steps:
    1. Verify the code meets all requirements
    2. Check for syntax errors or logical bugs
    3. Identify performance issues
    4. Suggest improvements or fixes
    5. Test the code if possible
    
    Be thorough but constructive in your feedback."""
    
    checking_model = st.session_state.checking_model
    
    combined_prompt = f"""User Request: {user_prompt}
    
    Planning Analysis:
    {planning_result}
    
    Code to Review:
    ```
    {code}
    ```
    
    Please review this code for correctness, potential bugs, and adherence to requirements. If possible, perform a virtual execution test and suggest improvements."""
    
    with st.spinner("Checking code..."):
        checking_result = query_openrouter(combined_prompt, checking_model, system_prompt, images)
    
    # Try to execute the code if it's Python
    if "python" in checking_result.lower() or "def " in code or "import " in code:
        try:
            result, error = execute_code(code)
            if error:
                execution_result = f"Execution Error: {error}"
            else:
                execution_result = f"Execution Output:\n{result}"
            st.session_state.execution_result = execution_result
        except Exception as e:
            st.session_state.execution_result = f"Execution Error: {str(e)}"
    
    # Move to the next task
    if st.session_state.current_task_index < len(st.session_state.tasks) - 1:
        st.session_state.current_task_index += 1
    
    return checking_result

# Function to handle image uploads
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            # For image files
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.session_state.uploaded_images.append(image)
                return True
            else:
                st.error("The uploaded file is not an image.")
                return False
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")
            return False
    return False

# Main UI
st.markdown("<h1 class='main-title'>AI Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your all-in-one AI assistant for planning, coding, and testing solutions</p>", unsafe_allow_html=True)

# Sidebar for settings
# API key input in main page
api_key = st.text_input(
    "Enter OpenRouter API Key",
    value=st.session_state.get('api_key', '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'),
    type="password",
    key="main_api_key"
)
st.session_state.api_key = api_key

# Create tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Code", "Task Progress"])

# Chat tab
with tab1:
    # Display conversation history
    # User input
    user_input = st.text_area("Type your message here:", key="chat_input", height=100)
    
    # Buttons
    plan_button = st.button("🧠 Plan", key="plan_btn")
    code_button = st.button("💻 Code", key="code_btn")
    check_button = st.button("✅ Check", key="check_btn")
    full_process_button = st.button("🎯 Full Process", key="full_process_btn")

    for i, entry in enumerate(st.session_state.conversation_history):
        if entry["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-content">
                    <strong>You:</strong><br>{entry["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display images if any
            if "images" in entry and entry["images"]:
                for img in entry["images"]:
                    st.image(img, use_column_width=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="avatar-container">🤖</div>
                <div class="message-content">
                    <strong>{entry["agent_type"]}:</strong><br>{entry["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="file_uploader")
    if uploaded_file:
        if process_uploaded_file(uploaded_file):
            st.success("Image uploaded successfully!")
            
    # Display currently uploaded images
    if st.session_state.uploaded_images:
        st.markdown("### Uploaded Images")
        cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        for i, img in enumerate(st.session_state.uploaded_images):
            cols[i % 3].image(img, use_column_width=True)
            
        if st.button("Clear Images"):
            st.session_state.uploaded_images = []
            st.success("Images cleared!")
    
    if user_input:  # Process input automatically when submitted
        handle_full_process(user_input, uploaded_file)
    
    # Handle button clicks
    if plan_button and user_input:
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "images": st.session_state.uploaded_images.copy() if st.session_state.uploaded_images else []
        })
        
        # Get planning result
        planning_result = plan_solution(user_input, st.session_state.uploaded_images)
        
        # Add planning result to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "agent_type": "Pineapple"
,
            "content": planning_result
        })
        
        # Rerun to update the UI
        st.rerun()
    
    elif code_button and user_input:
        # Check if we have a planning result
        planning_result = None
        for entry in reversed(st.session_state.conversation_history):
            if entry.get("agent_type") == "Planning Agent":
                planning_result = entry["content"]
                break
        
        if not planning_result:
            st.error("Please run the planning step first.")
        else:
            # Add user message to history if it's not already there
            if not any(entry["role"] == "user" and entry["content"] == user_input for entry in st.session_state.conversation_history[-2:]):
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "images": st.session_state.uploaded_images.copy() if st.session_state.uploaded_images else []
                })
            
            # Generate code
            coding_result, code = generate_code(user_input, planning_result, st.session_state.uploaded_images)
            
            # Add coding result to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "agent_type": "Coding Agent",
                "content": coding_result
            })
            
            # Rerun to update the UI
            st.rerun()
    
    elif check_button and user_input:
        # Check if we have coding and planning results
        planning_result = None
        code = st.session_state.code_output
        
        for entry in reversed(st.session_state.conversation_history):
            if entry.get("agent_type") == "Planning Agent" and not planning_result:
                planning_result = entry["content"]
        
        if not planning_result or not code:
            st.error("Please run the planning and coding steps first.")
        else:
            # Add user message to history if it's not already there
            if not any(entry["role"] == "user" and entry["content"] == user_input for entry in st.session_state.conversation_history[-3:]):
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "images": st.session_state.uploaded_images.copy() if st.session_state.uploaded_images else []
                })
            
            # Check code
            checking_result = check_code(user_input, planning_result, code, st.session_state.uploaded_images)
            
            # Add checking result to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "agent_type": "Checking Agent",
                "content": checking_result
            })
            
            # Rerun to update the UI
            st.rerun()
    
    elif full_process_button and user_input:
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "images": st.session_state.uploaded_images.copy() if st.session_state.uploaded_images else []
        })
        
        # Run full process
        # 1. Planning
        planning_result = plan_solution(user_input, st.session_state.uploaded_images)
        st.session_state.conversation_history.append({
            "role": "assistant",
            "agent_type": "Planning Agent",
            "content": planning_result
        })
        
        # 2. Coding
        coding_result, code = generate_code(user_input, planning_result, st.session_state.uploaded_images)
        st.session_state.conversation_history.append({
            "role": "assistant",
            "agent_type": "Coding Agent",
            "content": coding_result
        })
        
        # 3. Checking
        checking_result = check_code(user_input, planning_result, code, st.session_state.uploaded_images)
        st.session_state.conversation_history.append({
            "role": "assistant",
            "agent_type": "Checking Agent",
            "content": checking_result
        })
        
        # Rerun to update the UI
        st.rerun()

# Code tab
with tab2:
    if st.session_state.code_output:
        st.markdown("### Generated Code")
        
        # Display code with syntax highlighting
        st.code(st.session_state.code_output, line_numbers=True)
        
        # Copy button
        if st.button("Copy Code to Clipboard"):
            st.code(st.session_state.code_output)
            st.success("Code copied to clipboard! (Use Ctrl+C)")
        
        # Download button
        if st.download_button(
            label="Download Code",
            data=st.session_state.code_output,
            file_name="generated_code.py",
            mime="text/plain"
        ):
            st.success("Code downloaded successfully!")
        
        # Execution result
        if st.session_state.execution_result:
            st.markdown("### Execution Result")
            st.text_area("Output", value=st.session_state.execution_result, height=200, disabled=True)
    else:
        st.info("No code has been generated yet. Use the Chat tab to generate code.")

# Task Progress tab
with tab3:
    st.markdown("### Task Progress")
    
    if st.session_state.tasks:
        progress_container = st.container()
        
        with progress_container:
            st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
            
            for i, task in enumerate(st.session_state.tasks):
                if i < st.session_state.current_task_index:
                    st.markdown(f"✅ <span class='completed-task'>{task}</span>", unsafe_allow_html=True)
                elif i == st.session_state.current_task_index:
                    st.markdown(f"▶️ <span class='current-task'>{task}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"⏳ <span class='future-task'>{task}</span>", unsafe_allow_html=True)
            
            progress_percentage = (st.session_state.current_task_index / max(1, len(st.session_state.tasks) - 1)) * 100
            progress_bar = st.progress(0)  # Initialize progress bar
            progress_bar.progress(progress_percentage / 100)

            # Update status text
            status_text.text(f"Progress: {int(progress_percentage)}% - {st.session_state.tasks[st.session_state.current_task_index]}")
            
            # Create a placeholder for showing the current output
            output_placeholder = st.empty()
            
            # Execute the workflow
            if st.session_state.current_task_index == 0:  # Planning
                with st.spinner("Planning in progress..."):
                    planning_output = call_agent("planning", user_request)
                    st.session_state.planning_output = planning_output
                    output_placeholder.markdown(f"### Planning Output\n```\n{planning_output}\n```")
                    st.session_state.current_task_index += 1
            
            elif st.session_state.current_task_index == 1:  # Coding
                with st.spinner("Coding in progress..."):
                    coding_prompt = f"""
    USER REQUEST: {user_request}
    
    PLANNING DETAILS:
    {st.session_state.planning_output}
    
    Based on the planning details above, please implement the complete code solution.
    """
                    coding_output = call_agent("coding", coding_prompt)
                    st.session_state.coding_output = coding_output
                    
                    # Extract code from the coding output
                    st.session_state.extracted_code = extract_code(coding_output)
                    
                    output_placeholder.markdown(f"### Coding Output\n```python\n{st.session_state.extracted_code}\n```")
                    st.session_state.current_task_index += 1
            
            elif st.session_state.current_task_index == 2:  # Checking
                with st.spinner("Checking code..."):
                    checking_prompt = f"""
    USER REQUEST: {user_request}
    
    PLANNING DETAILS:
    {st.session_state.planning_output}
    
    CODE IMPLEMENTATION:
    {st.session_state.extracted_code}
    
    Please review the code, identify any issues, and test it if possible. Provide detailed feedback.
    """
                    checking_output = call_agent("checking", checking_prompt)
                    st.session_state.checking_output = checking_output
                    
                    output_placeholder.markdown(f"### Code Review\n```\n{checking_output}\n```")
                    st.session_state.current_task_index += 1
            
            elif st.session_state.current_task_index == 3:  # Final Result
                final_code = st.session_state.extracted_code
                
                # Check if there were issues from the checking output
                if "FAIL" in st.session_state.checking_output or "fail" in st.session_state.checking_output.lower():
                    with st.spinner("Fixing code based on feedback..."):
                        fixing_prompt = f"""
    USER REQUEST: {user_request}
    
    PLANNING DETAILS:
    {st.session_state.planning_output}
    
    ORIGINAL CODE:
    {st.session_state.extracted_code}
    
    CODE REVIEW FEEDBACK:
    {st.session_state.checking_output}
    
    Please fix the code based on the review feedback. Provide the complete corrected code.
    """
                        fixing_output = call_agent("coding", fixing_prompt)
                        final_code = extract_code(fixing_output)
                        
                        st.session_state.final_code = final_code
                        output_placeholder.markdown(f"### Fixed Code\n```python\n{final_code}\n```")
                else:
                    st.session_state.final_code = final_code
                    output_placeholder.markdown(f"### Final Code (Passed Review)\n```python\n{final_code}\n```")
                
                st.session_state.current_task_index += 1
            
            elif st.session_state.current_task_index == 4:  # Complete
                # Display final output
                st.success("Workflow completed successfully!")
                
                # Display the final code with a download button
                st.markdown("### Final Result")
                st.code(st.session_state.final_code, language="python")
                
                # Generate download link for the code
                filename = st.text_input("Filename", value="output.py")
                
                # Button to download the code
                st.download_button(
                    label="Download Code",
                    data=st.session_state.final_code,
                    file_name=filename,
                    mime="text/plain"
                )
                
                # Create a button to test the code
                if st.button("Test Code"):
                    test_code(st.session_state.final_code)

    # Function to test the code
    def test_code(code_content):
        """Test the provided code in a virtual environment."""
        try:
            st.markdown("### Code Execution Output")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                f.write(code_content)
                temp_file_path = f.name
            
            # Run the code and capture output
            with st.spinner("Running code..."):
                # Create a subprocess and capture output
                process = subprocess.Popen(
                    [sys.executable, temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Create containers for output
                stdout_container = st.empty()
                stderr_container = st.empty()
                
                stdout_output = ""
                stderr_output = ""
                
                # Read output in real-time
                while True:
                    stdout_line = process.stdout.readline()
                    stderr_line = process.stderr.readline()
                    
                    if not stdout_line and not stderr_line and process.poll() is not None:
                        break
                    
                    if stdout_line:
                        stdout_output += stdout_line
                        stdout_container.code(stdout_output)
                    
                    if stderr_line:
                        stderr_output += stderr_line
                        stderr_container.error(stderr_output)
                
                # Wait for the process to complete
                return_code = process.wait()
                
                # Cleanup the temp file
                os.unlink(temp_file_path)
                
                if return_code == 0:
                    st.success(f"Code executed successfully (exit code {return_code})")
                else:
                    st.error(f"Code execution failed with exit code {return_code}")
        
        except Exception as e:
            st.error(f"Error executing code: {str(e)}")

    # Function to call OpenRouter API
    def call_agent(agent_type, prompt, retries=0):
        """Call the specified agent type with the given prompt."""
        if agent_type == "planning":
            model_list = CONFIG["planning_agents"]
            index = st.session_state.current_planning_index
            system_prompt = SYSTEM_PROMPTS["planning"]
        elif agent_type == "coding":
            model_list = CONFIG["coding_agents"]
            index = st.session_state.current_coding_index
            system_prompt = SYSTEM_PROMPTS["coding"]
        else:  # checking
            model_list = CONFIG["checking_agents"]
            index = st.session_state.current_checking_index
            system_prompt = SYSTEM_PROMPTS["checking"]
        
        # Get the current model
        model = model_list[index]["id"]
        
        # Prepare the payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": CONFIG["temperature"],
            "max_tokens": CONFIG["max_tokens"]
        }
        
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            with st.spinner(f"Calling {model_list[index]['name']}..."):
                response = requests.post(
                    st.session_state.api_base,
                    headers=headers,
                    json=payload,
                    timeout=60  # Set a timeout for the request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    # If we get an error, try with the next model
                    if retries < len(model_list) - 1:
                        # Update the current index for this agent type
                        if agent_type == "planning":
                            st.session_state.current_planning_index = (st.session_state.current_planning_index + 1) % len(model_list)
                        elif agent_type == "coding":
                            st.session_state.current_coding_index = (st.session_state.current_coding_index + 1) % len(model_list)
                        else:  # checking
                            st.session_state.current_checking_index = (st.session_state.current_checking_index + 1) % len(model_list)
                        
                        # Retry with the next model
                        st.warning(f"Error with {model_list[index]['name']}. Retrying with next model...")
                        time.sleep(1)  # Brief pause before retry
                        return call_agent(agent_type, prompt, retries + 1)
                    else:
                        error_msg = f"All {agent_type} models failed. Status code: {response.status_code}, Response: {response.text}"
                        st.error(error_msg)
                        raise Exception(error_msg)
        except Exception as e:
            if retries < len(model_list) - 1:
                # Same retry logic as above
                if agent_type == "planning":
                    st.session_state.current_planning_index = (st.session_state.current_planning_index + 1) % len(model_list)
                elif agent_type == "coding":
                    st.session_state.current_coding_index = (st.session_state.current_coding_index + 1) % len(model_list)
                else:  # checking
                    st.session_state.current_checking_index = (st.session_state.current_checking_index + 1) % len(model_list)
                
                st.warning(f"Error with {model_list[index]['name']}. Retrying with next model...")
                time.sleep(1)
                return call_agent(agent_type, prompt, retries + 1)
            else:
                error_message = f"All {agent_type} models failed with error: {str(e)}"
                st.error(error_message)
                raise Exception(error_message)

    def extract_code(text):
        """Extract code blocks from text."""
        # Look for code blocks surrounded by triple backticks
        code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)\s*```', text)
        
        if code_blocks:
            # Join multiple code blocks if found
            return '\n\n'.join(code_blocks)
        
        # If no code blocks found, look for sections labeled as code
        code_section = re.search(r'CODE IMPLEMENTATION:\s*([\s\S]*?)(?:\n\n|$)', text)
        if code_section:
            return code_section.group(1).strip()
        
        # If still no code found, just return the text as is
        return text

    # Image upload and handling
    def handle_image_upload():
        """Handle the image upload and return the base64 encoded image."""
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Read the file and encode it
            bytes_data = uploaded_file.getvalue()
            encoded = base64.b64encode(bytes_data).decode()
            
            # Display the image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            return encoded
        return None

    # Initialize session state variables
    if 'planning_output' not in st.session_state:
        st.session_state.planning_output = ""
    if 'coding_output' not in st.session_state:
        st.session_state.coding_output = ""
    if 'checking_output' not in st.session_state:
        st.session_state.checking_output = ""
    if 'extracted_code' not in st.session_state:
        st.session_state.extracted_code = ""
    if 'final_code' not in st.session_state:
        st.session_state.final_code = ""
    if 'tasks' not in st.session_state:
        st.session_state.tasks = ["Planning", "Coding", "Checking", "Final Result", "Complete"]
    if 'current_task_index' not in st.session_state:
        st.session_state.current_task_index = 0
    if 'image_data' not in st.session_state:
        st.session_state.image_data = None

    # Page config is already set at the beginning of the script

    # Apply custom CSS for styling
    st.markdown("""
    <style>
    /* Dark theme */
    body {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    .stTextArea > div > div > textarea {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 15px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #7EB4EA;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2D2D2D;
        border-radius: 4px 4px 0 0;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set the title with a logo
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌲 PineFile</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #7EB4EA;'>An AI Development Assistant</h3>", unsafe_allow_html=True)

    # Sidebar
    # Set default models and API key input in main page
    st.session_state.api_key = st.session_state.get('api_key', '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')
    planning_model = "anthropic/claude-3.5-sonnet"
    coding_model = "anthropic/claude-3.5-sonnet"
    checking_model = "anthropic/claude-3.5-sonnet"

    # Main content area
    tabs = st.tabs(["Development", "Instructions", "About"])

    # Development tab
    with tabs[0]:
        # User input section
        st.markdown("### 📝 Your Request")
        user_request = st.text_area(
            "Describe what you want to build:",
            height=150,
            placeholder="Example: Create a simple weather app that retrieves data from an API and displays the current weather and forecast for a given city."
        )
        
        # Image upload option
        with st.expander("Add an image"):
            st.session_state.image_data = handle_image_upload()
            if st.session_state.image_data:
                st.info("Image uploaded successfully. It will be included in your request.")
        
        # Adjust the user request if there's an image
        if user_request and st.session_state.image_data:
            user_request += f"\n\n[IMAGE: {st.session_state.image_data}]"
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit_button = st.button("🚀 Submit", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            for key in ['planning_output', 'coding_output', 'checking_output', 'extracted_code', 'final_code', 'current_task_index', 'image_data']:
                if key in st.session_state:
                    st.session_state[key] = "" if key != 'current_task_index' else 0
            st.experimental_rerun()
        
        # Display progress
        st.markdown("### 📊 Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create task checklist
        st.markdown("### ✅ Tasks")
        for i, task in enumerate(st.session_state.tasks):
            if i < st.session_state.current_task_index:
                st.markdown(f"- [x] {task} ✅")
            elif i == st.session_state.current_task_index:
                st.markdown(f"- [ ] {task} 🔄")
            else:
                st.markdown(f"- [ ] {task}")
        
        st.markdown("---")
        
        # Output area
        st.markdown("### 📄 Output")
        
        # Show outputs in their respective sections based on current state
        output_tabs = st.tabs(["Planning", "Coding", "Checking", "Final Result"])
        
        with output_tabs[0]:
            if st.session_state.planning_output:
                st.markdown("#### Planning Output")
                st.text_area("", value=st.session_state.planning_output, height=300, disabled=True)
        
        with output_tabs[1]:
            if st.session_state.coding_output:
                st.markdown("#### Coding Output")
                st.code(st.session_state.extracted_code, language="python")
        
        with output_tabs[2]:
            if st.session_state.checking_output:
                st.markdown("#### Code Review")
                st.text_area("", value=st.session_state.checking_output, height=300, disabled=True)
        
        with output_tabs[3]:
            if st.session_state.final_code:
                st.markdown("#### Final Code")
                st.code(st.session_state.final_code, language="python")
                
                # Download button
                filename = st.text_input("Filename", value="output.py")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Code",
                        data=st.session_state.final_code,
                        file_name=filename,
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("🔄 Test Code", use_container_width=True):
                        test_code(st.session_state.final_code)
        
        # Process the request when submitted
        if submit_button and user_request:
            # Reset progress
            st.session_state.current_task_index = 0
            
            # Clear previous outputs
            for key in ['planning_output', 'coding_output', 'checking_output', 'extracted_code', 'final_code']:
                st.session_state[key] = ""
            
            # Update the page to show the workflow has started
            st.experimental_rerun()

    # Instructions tab
    with tabs[1]:
        st.markdown("""
        ## 📚 How to Use PineFile
        
        PineFile is an AI-powered development assistant that helps you build software by breaking down the development process into three key stages:
        
        ### 1. Planning 🧠
        The planning agent analyzes your request, researches APIs and libraries, and develops a structured approach for implementation.
        
        ### 2. Coding 💻
        The coding agent takes the planning output and creates functional code that implements your requirements.
        
        ### 3. Checking ✅
        The checking agent reviews the code for bugs, efficiency issues, and adherence to best practices, then tests it when possible.
        
        ### Getting Started
        
        1. **Enter your request** in the text area, describing what you want to build.
        2. Optionally, **upload an image** if it helps explain your request.
        3. Click the **Submit** button to start the process.
        4. Follow the progress as each agent works on your request.
        5. When complete, you can download or test the final code.
        
        ### Tips for Better Results
        
        - Be specific about what you want the code to do.
        - Mention programming languages, frameworks, or libraries you prefer.
        - Specify any constraints or requirements (performance, compatibility, etc.).
        - For complex projects, consider breaking them down into smaller requests.
        
        ### Troubleshooting
        
        - If an agent fails, PineFile will automatically try alternative models.
        - If all models fail, check your API key and internet connection.
        - For persistent issues, try simplifying your request or breaking it into smaller parts.
        """)

    # About tab
    with tabs[2]:
        st.markdown("""
        ## 🌲 About PineFile
        
        PineFile is an AI development assistant that leverages multiple AI models through OpenRouter to help you build software applications.
        
        ### Features
        
        - **Multi-agent workflow**: Uses specialized AI models for planning, coding, and checking.
        - **Fallback mechanism**: Automatically switches to alternative models if one fails.
        - **Code testing**: Runs your code in a sandboxed environment to verify functionality.
        - **Easy export**: Download your completed code with a single click.
        - **Image support**: Include images to better explain your requirements.
        
        ### Models Used
        
        PineFile uses free models available through OpenRouter:
        
        **Planning Agents:**
        - OpenAI Mixtral 8x7B
        - Anthropic Claude 2
        - OpenAI GPT-3.5 Turbo
        - Google PaLM 2
        - Meta Llama 3 70B
        
        **Coding Agents:**
        - Codestral
        - DeepSeek Coder
        - Phind CodeLlama
        - WizardCoder
        - Code Llama
        
        **Checking Agents:**
        - DeepSeek Coder
        - OpenAI GPT-3.5 Turbo
        - Meta Llama 3 70B
        - Mistral Large
        - Claude 2
        
        ### Credits
        
        Built with:
        - Python
        - Streamlit
        - OpenRouter API
        """)

# Run the main app
if __name__ == "__main__":
    # Since we're not using a main() function, we don't need to call anything here
    # All the Streamlit code runs at module level
    pass