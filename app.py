import gradio as gr
import json
import os
import tempfile
from src.pipeline.dubbing import VideoDubber

# Global instance to keep model loaded
dubber = None

def get_dubber():
    global dubber
    if dubber is None:
        print("Initializing VideoDubber...")
        dubber = VideoDubber()
    return dubber

def generate_audio(script_json_str):
    try:
        script = json.loads(script_json_str)
    except json.JSONDecodeError as e:
        return None, f"JSON Error: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"

    if not isinstance(script, list):
        return None, "Error: Script must be a JSON list of objects."

    # Create a temp file for output
    output_dir = "web_outputs"
    os.makedirs(output_dir, exist_ok=True)
    # Use a fixed name or unique name? Fixed name is easier for overwrite, unique is better for multi-user (but this is local)
    # Let's use fixed name to save space for this demo
    output_path = os.path.join(output_dir, "generated_dubbing.wav")

    try:
        dubber_instance = get_dubber()
        # The dubber returns the path
        result_path = dubber_instance.generate_audio_track(script, output_path)
        return result_path, "Success! Audio generated."
    except Exception as e:
        return None, f"Generation Error: {str(e)}"

# Default demo script
default_script = """[
  {
    "start": 0.0,
    "text": "欢迎使用 Vox Timeline 配音系统。",
    "speaker": "Uncle_Fu",
    "instruct": "开心"
  },
  {
    "start": 3.0,
    "text": "这是一个演示脚本，用于测试从文本直接生成时间轴音频的功能。",
    "speaker": "Uncle_Fu",
    "instruct": "认真"
  }
]"""

with gr.Blocks(title="Vox Timeline Web UI") as app:
    gr.Markdown("# Vox Timeline - AI Video Dubbing System")
    gr.Markdown("Enter your dubbing script in JSON format below and click Generate.")
    
    with gr.Row():
        with gr.Column():
            script_input = gr.Code(value=default_script, language="json", label="Dubbing Script (JSON)")
            generate_btn = gr.Button("Generate Audio", variant="primary")
        
        with gr.Column():
            status_output = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="filepath", interactive=False)

    generate_btn.click(
        fn=generate_audio,
        inputs=[script_input],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    # Launch on 127.0.0.1
    app.launch(server_name="127.0.0.1")
