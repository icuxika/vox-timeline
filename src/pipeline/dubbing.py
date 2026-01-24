import json
import os
from typing import List, Dict, Optional
from src.core.tts import TTSEngine
from src.core.audio import AudioTimeline
from moviepy import VideoFileClip, AudioFileClip
import soundfile as sf

class VideoDubber:
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device: str = "cuda:0"):
        self.tts = TTSEngine(model_name, device)
    
    def load_script(self, script_path: str) -> List[Dict]:
        with open(script_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_audio_track(self, script: List[Dict], output_path: str, debug_dir: Optional[str] = None, 
                             default_speaker: str = "Uncle_Fu", default_language: str = "Chinese",
                             total_duration: Optional[float] = None):
        timeline = AudioTimeline()
        
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug mode enabled: saving segments to {debug_dir}")

        print(f"Processing {len(script)} segments with speaker={default_speaker}, language={default_language}...")
        for i, segment in enumerate(script):
            start_time = segment.get("start", 0.0)
            text = segment.get("text", "")
            # Override script settings with global defaults if provided, 
            # OR strictly enforce global defaults as requested by user.
            # User said: "a complete video can only contain one speaker+language selection"
            # So we strictly use the passed in arguments.
            speaker = default_speaker
            language = default_language
            instruct = segment.get("instruct", None)
            
            if not text:
                continue
                
            print(f"[{i+1}/{len(script)}] Generating at {start_time}s: {text[:20]}...")
            try:
                audio_data, sr = self.tts.generate(text, speaker=speaker, language=language, instruct=instruct)
                
                # Debug: save individual segment
                if debug_dir:
                    filename = f"seg_{i:03d}_{start_time}s.wav"
                    filepath = os.path.join(debug_dir, filename)
                    sf.write(filepath, audio_data, sr)
                    print(f"  -> Saved debug file: {filepath}")

                timeline.add_segment(start_time, audio_data, sr)
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
        
        # Convert total_duration to ms if provided
        target_duration_ms = int(total_duration * 1000) if total_duration else None
        timeline.export(output_path, target_duration_ms=target_duration_ms)
        return output_path

    def dub_video(self, video_path: str, audio_path: str, output_path: str):
        """
        Replace video audio with the generated audio track.
        """
        print(f"Loading video: {video_path}")
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(audio_path)
        
        # If new audio is shorter/longer, handle it? 
        # For now, let's just set the audio.
        # Note: If audio is longer than video, video will be extended or audio cut?
        # Usually we want the video length to be maintained or audio length.
        
        final_video = video.with_audio(new_audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Video saved to {output_path}")

