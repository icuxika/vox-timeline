import gradio as gr
import json
import os
import tempfile
from src.pipeline.dubbing import VideoDubber
from src.pipeline.video_translator import VideoTranslatorPipeline

# Global instance to keep model loaded
dubber = None
translator_pipeline = None

def get_dubber():
    global dubber
    if dubber is None:
        print("Initializing VideoDubber...")
        dubber = VideoDubber()
    return dubber

def get_translator():
    global translator_pipeline
    if translator_pipeline is None:
        print("Initializing VideoTranslatorPipeline...")
        translator_pipeline = VideoTranslatorPipeline()
    return translator_pipeline

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

TRANSLATION_LANG_MAP = {
    "Chinese": "zh",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru"
}

def translate_video(video_file, source_lang_choice, target_lang_choice, speaker_choice, subtitle_mode_choice):
    if not video_file:
        return None, None, None, None, None, "Error: Please upload a video file."
        
    try:
        pipeline = get_translator()
        
        # Map friendly name to code
        target_code = TRANSLATION_LANG_MAP.get(target_lang_choice, "en")
        source_code = TRANSLATION_LANG_MAP.get(source_lang_choice, "auto")
        if source_lang_choice == "Auto":
            source_code = "auto"
            
        # Map subtitle mode choice
        subtitle_mode = "soft" if "Soft" in subtitle_mode_choice else "hard"
            
        output_dir = "web_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # video_file is a file path in Gradio 4.x
        final_audio, script, src_srt, trans_srt, final_video = pipeline.process_video(
            video_path=video_file,
            source_lang=source_code,
            target_lang=target_code,
            output_dir=output_dir,
            speaker=speaker_choice,
            subtitle_mode=subtitle_mode
        )
        
        script_json = json.dumps(script, ensure_ascii=False, indent=2)
        return final_audio, script_json, src_srt, trans_srt, final_video, f"Success! Video translated to {target_lang_choice} (Subtitles: {subtitle_mode})."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, f"Translation Error: {str(e)}"

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
    
    with gr.Tab("📝 Script Dubbing (脚本配音)"):
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

    with gr.Tab("🎥 Video Translation (视频翻译配音)"):
        gr.Markdown("上传视频，系统将自动提取音频、识别字幕、翻译并生成新的配音。")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video (上传视频)")
                
                with gr.Row():
                    trans_source_lang = gr.Dropdown(choices=LANGUAGE_OPTIONS, value="Auto", label="Source Language (源语言)")
                    trans_target_lang = gr.Dropdown(choices=[l for l in LANGUAGE_OPTIONS if l != "Auto"], value="Chinese", label="Target Language (目标语言)")
                
                trans_speaker = gr.Dropdown(choices=SPEAKER_OPTIONS, value="uncle_fu", label="Select Speaker (选择配音员)")
                
                trans_subtitle_mode = gr.Radio(choices=["Hard Subtitles (硬字幕)", "Soft Subtitles (软字幕)"], value="Hard Subtitles (硬字幕)", label="Subtitle Type (字幕类型)")
                
                translate_btn = gr.Button("🌍 Translate & Dub (翻译并配音)", variant="primary")
                
            with gr.Column(scale=1):
                trans_status = gr.Textbox(label="Status", interactive=False)
                trans_video_output = gr.Video(label="Final Translated Video (最终视频)", interactive=False)
                trans_audio_output = gr.Audio(label="Translated Audio (翻译后音频)", type="filepath", interactive=False)
                
                with gr.Row():
                    src_srt_output = gr.File(label="Original Subtitles (原文字幕)", interactive=False)
                    trans_srt_output = gr.File(label="Translated Subtitles (译文字幕)", interactive=False)
                    
                trans_script_output = gr.Code(language="json", label="Generated Script (生成脚本)", interactive=False)
            
            translate_btn.click(
                fn=translate_video,
                inputs=[video_input, trans_source_lang, trans_target_lang, trans_speaker, trans_subtitle_mode],
                outputs=[trans_audio_output, trans_script_output, src_srt_output, trans_srt_output, trans_video_output, trans_status]
            )

if __name__ == "__main__":
    # Launch on 127.0.0.1
    app.launch(server_name="127.0.0.1")
