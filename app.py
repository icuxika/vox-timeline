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

def generate_audio(script_json_str, speaker_choice, language_choice):
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
    output_path = os.path.join(output_dir, "generated_dubbing.wav")

    try:
        dubber_instance = get_dubber()
        # The dubber returns the path
        result_path = dubber_instance.generate_audio_track(
            script, 
            output_path, 
            default_speaker=speaker_choice, 
            default_language=language_choice
        )
        return result_path, f"Success! Audio generated with Speaker: {speaker_choice}, Language: {language_choice}"
    except Exception as e:
        return None, f"Generation Error: {str(e)}"

# Speaker Data Reference (Keep for user info, but selection is now via Dropdown)
SPEAKER_INFO = """
### 🎙️ 配音角色参考 (Speaker Reference)

| Speaker | Voice Description (声音描述) | Native Language (母语) |
| :--- | :--- | :--- |
| **vivian** | 明亮、略带棱角的年轻女性声音 | Chinese |
| **serena** | 温暖、温柔的年轻女性声音 | Chinese |
| **uncle_fu** | 醇厚、低沉的成熟男性声音 | Chinese |
| **dylan** | 青春、清脆自然的北京口音男性声音 | Chinese (Beijing) |
| **eric** | 活泼、略带沙哑明亮的成都口音男性声音 | Chinese (Sichuan) |
| **ryan** | 充满活力、节奏感强的男性声音 | English |
| **aiden** | 阳光、中频清晰的美国男性声音 | English |
| **ono_anna** | 俏皮、轻盈灵动的日语女性声音 | Japanese |
| **sohee** | 温暖、情感丰富的韩语女性声音 | Korean |
"""

SPEAKER_OPTIONS = [
    "aiden", "dylan", "eric", "ono_anna", "ryan", 
    "serena", "sohee", "uncle_fu", "vivian"
]

LANGUAGE_OPTIONS = [
    "Auto", "Chinese", "English", "French", "German", "Italian", 
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish"
]

# Default demo script (Removed per-segment speaker/language)
default_script = """[
  {
    "start": 0.0,
    "text": "欢迎使用 Vox Timeline 配音系统。",
    "instruct": "开心"
  },
  {
    "start": 3.0,
    "text": "现在整个视频将统一使用您选择的说话人和语言。",
    "instruct": "认真"
  },
  {
    "start": 6.0,
    "text": "无论您选择什么语言，我都会尝试用该语言朗读这些文本。",
    "instruct": "excited"
  }
]"""

with gr.Blocks(title="Vox Timeline Web UI") as app:
    gr.Markdown("# Vox Timeline - AI Video Dubbing System")
    gr.Markdown("输入 JSON 格式的配音脚本，并在右侧选择全局 **Speaker** 和 **Language**，系统将为整个视频生成统一风格的配音。")
    
    with gr.Row():
        with gr.Column(scale=2):
            script_input = gr.Code(value=default_script, language="json", label="Dubbing Script (JSON)")
            
            with gr.Accordion("📚 查看角色详情 (Speaker Details)", open=False):
                gr.Markdown(SPEAKER_INFO)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ 全局设置 (Global Settings)")
            
            # Global controls
            speaker_dropdown = gr.Dropdown(choices=SPEAKER_OPTIONS, value="uncle_fu", label="Select Speaker (选择说话人)")
            language_dropdown = gr.Dropdown(choices=LANGUAGE_OPTIONS, value="Chinese", label="Select Language (选择语言)")
            
            generate_btn = gr.Button("🎵 Generate Audio (生成音频)", variant="primary")
            
            status_output = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="filepath", interactive=False)

    generate_btn.click(
        fn=generate_audio,
        inputs=[script_input, speaker_dropdown, language_dropdown],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    # Launch on 127.0.0.1
    app.launch(server_name="127.0.0.1")
