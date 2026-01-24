import os
import json
import subprocess
from typing import Optional, List, Dict, Tuple
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback for newer moviepy versions or if structure changed
    from moviepy.video.io.VideoFileClip import VideoFileClip

from src.core.asr import WhisperASR
from src.core.translator import TranslateGemma, HelsinkiOpusTranslator
from src.pipeline.dubbing import VideoDubber

class VideoTranslatorPipeline:
    def __init__(self):
        self.asr = None
        self.translators = {} # Cache translator instances
        self.dubber = VideoDubber()
        
        self.lang_map = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian"
        }

    def _get_translator(self, choice: str = "gemma"):
        if choice not in self.translators:
            if choice == "gemma":
                self.translators[choice] = TranslateGemma()
            elif choice == "helsinki":
                self.translators[choice] = HelsinkiOpusTranslator()
            else:
                # Default to gemma
                self.translators[choice] = TranslateGemma()
        return self.translators[choice]

    def _ensure_models_loaded(self):
        if not self.asr:
            self.asr = WhisperASR()
        # Translators are loaded on demand via _get_translator

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        millis = int((seconds % 1) * 1000)
        seconds = int(seconds)
        minutes = seconds // 60
        hours = minutes // 60
        minutes = minutes % 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _generate_srt(self, segments: List[Dict], output_path: str, text_key: str = "text"):
        """Generate SRT file from segments"""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                start = self._format_time(seg["start"])
                end = self._format_time(seg["end"])
                text = seg.get(text_key, "").strip()
                
                f.write(f"{i+1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        print(f"Generated subtitles: {output_path}")

    def process_video(self, video_path: str, source_lang: str, target_lang: str, 
                      output_dir: str = "output", speaker: str = "uncle_fu",
                      subtitle_mode: str = "hard", translator_choice: str = "gemma") -> Tuple[str, List[Dict], str, str, str]:
        """
        Process video: Extract Audio -> ASR -> Translate -> TTS -> Merge
        
        Args:
            video_path: Path to input video.
            source_lang: Source language code (e.g., 'zh', 'en').
            target_lang: Target language code (e.g., 'en', 'zh').
            output_dir: Directory to save outputs.
            speaker: TTS speaker to use.
            subtitle_mode: "hard" (burn-in) or "soft" (mov_text stream).
            translator_choice: "gemma" or "helsinki".
            
        Returns:
            Tuple of (final_audio_path, dubbing_script, src_srt_path, trans_srt_path, final_video_path)
        """
        self._ensure_models_loaded()
        translator = self._get_translator(translator_choice)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Extract Audio
        print(f"Processing video: {video_path}")
        audio_path = os.path.join(output_dir, "temp_extracted.wav")
        
        try:
            video = VideoFileClip(video_path)
            video_duration = video.duration # Get video duration in seconds
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {e}")
        
        # 2. ASR (Whisper)
        print(f"Running ASR (Source: {source_lang})...")
        # Whisper can auto-detect, but providing language helps accuracy if known
        asr_lang = source_lang if source_lang != "auto" else None
        segments = self.asr.transcribe(audio_path, language=asr_lang)
        
        # 3. Translate (Gemma)
        print(f"Translating to {target_lang}...")
        dubbing_script = []
        
        # If source is auto, we default to 'en' for translation if we can't detect it easily from simple ASR wrapper
        # TODO: Enhance ASR wrapper to return detected language
        trans_source_lang = source_lang if source_lang != "auto" else "en"
        
        for i, seg in enumerate(segments):
            original_text = seg["text"].strip()
            if not original_text:
                continue
                
            try:
                translated_text = translator.translate(
                    original_text, 
                    source_lang=trans_source_lang, 
                    target_lang=target_lang
                )
                print(f"[{i}] {original_text} -> {translated_text}")
                
                dubbing_script.append({
                    "start": seg["start"],
                    "end": seg["end"], # Added for SRT generation
                    "text": translated_text,
                    "instruct": "neutral" # Default instruction
                })
            except Exception as e:
                print(f"Translation failed for segment {i}: {e}")
                # Fallback to original text or skip? Let's skip to avoid mixing languages badly
                continue
            
        # Save script for debugging
        script_path = os.path.join(output_dir, "translated_script.json")
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(dubbing_script, f, ensure_ascii=False, indent=2)
            
        # Generate Subtitles (SRT)
        src_srt_path = os.path.join(output_dir, "original.srt")
        trans_srt_path = os.path.join(output_dir, "translated.srt")
        
        self._generate_srt(segments, src_srt_path, text_key="text")
        self._generate_srt(dubbing_script, trans_srt_path, text_key="text")

        # 4. Generate Audio (TTS)
        print("Generating dubbing audio...")
        final_audio_path = os.path.join(output_dir, "final_dubbed.wav")
        
        tts_lang = self.lang_map.get(target_lang, "English")
        
        self.dubber.generate_audio_track(
            script=dubbing_script,
            output_path=final_audio_path,
            default_speaker=speaker,
            default_language=tts_lang,
            total_duration=video_duration
        )
        
        # 5. Merge Video + Audio + Subtitles
        print(f"Merging video, audio, and subtitles (Mode: {subtitle_mode})...")
        final_video_path = os.path.join(output_dir, "final_translated_video.mp4")
        
        # Use relative paths for ffmpeg to avoid windows path escaping issues in filter
        # We need to change cwd to output_dir temporarily or just use absolute path but handle escaping
        # Actually, using forward slashes with absolute path often works in ffmpeg filters on Windows too,
        # but relative path is safer if we run from project root.
        # Let's try to convert absolute path to forward slashes.
        abs_srt_path = os.path.abspath(trans_srt_path).replace("\\", "/")
        # Escape colons for ffmpeg filter: C:/ -> C\:/
        escaped_srt_path = abs_srt_path.replace(":", "\\:")
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", final_audio_path]

        if subtitle_mode == "soft":
            # Soft subtitles logic (mov_text)
            # -c:v copy (fast)
            # -c:a aac
            # -c:s mov_text
            # Inputs: 0:video, 1:audio, 2:srt
            cmd.extend(["-i", trans_srt_path])
            cmd.extend([
                "-map", "0:v",
                "-map", "1:a",
                "-map", "2:s",
                "-c:v", "copy",
                "-c:a", "aac",
                "-c:s", "mov_text",
                "-shortest",
                final_video_path
            ])
        else:
            # Hard subtitles logic (burn-in)
            # -vf subtitles=...
            # -c:v libx264 (re-encode)
            # -c:a aac
            cmd.extend([
                "-map", "0:v",
                "-map", "1:a",
                "-vf", f"subtitles='{escaped_srt_path}':force_style='Fontsize=20'",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                final_video_path
            ])
        
        print(f"Executing FFmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        return final_audio_path, dubbing_script, src_srt_path, trans_srt_path, final_video_path
