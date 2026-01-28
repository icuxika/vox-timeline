import os
import json
import subprocess
import time
import re
from typing import Optional, List, Dict, Tuple, Iterator, Union
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback for newer moviepy versions or if structure changed
    from moviepy.video.io.VideoFileClip import VideoFileClip

from src.core.asr import WhisperASR
from src.core.translator import TranslateGemma, HelsinkiOpusTranslator, HymtTranslator
from src.pipeline.dubbing import VideoDubber

def format_remaining(seconds):
    if seconds is None or seconds < 0: return "..."
    if seconds < 60: return f"{int(seconds)}s"
    return f"{int(seconds)//60}m {int(seconds)%60}s"

class VideoTranslatorPipeline:
    def __init__(self):
        self.asr = None
        self.translators = {} # Cache translator instances
        self.dubber = VideoDubber()
        self.cancel_flag = False
        
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

    def cancel_task(self):
        """Set cancel flag to stop processing"""
        self.cancel_flag = True

    def _get_translator(self, choice: str = "gemma"):
        if choice not in self.translators:
            if choice == "gemma":
                self.translators[choice] = TranslateGemma()
            elif choice == "helsinki":
                self.translators[choice] = HelsinkiOpusTranslator()
            elif choice == "hymt":
                self.translators[choice] = HymtTranslator()
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
                      subtitle_mode: str = "hard", translator_choice: str = "gemma",
                      dubbing_enabled: bool = True) -> Iterator[Union[Tuple[str, float, str], Tuple[str, Tuple[str, List[Dict], str, str, str]]]]:
        """
        Process video: Extract Audio -> ASR -> Translate -> TTS -> Merge
        Yields progress updates: ("progress", float, "message")
        Yields final result: ("result", (final_audio_path, dubbing_script, src_srt_path, trans_srt_path, final_video_path))
        """
        self._ensure_models_loaded()
        self.cancel_flag = False # Reset flag
        translator = self._get_translator(translator_choice)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Extract Audio
        yield ("progress", 0.05, f"Extracting audio from {os.path.basename(video_path)}...")
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
        yield ("progress", 0.15, f"Running ASR (Source: {source_lang})...")
        print(f"Running ASR (Source: {source_lang})...")
        # Whisper can auto-detect, but providing language helps accuracy if known
        asr_lang = source_lang if source_lang != "auto" else None
        segments = self.asr.transcribe(audio_path, language=asr_lang)
        
        # 3. Translate (Gemma)
        yield ("progress", 0.30, f"Translating to {target_lang}...")
        print(f"Translating to {target_lang}...")
        dubbing_script = []
        
        # If source is auto, we default to 'en' for translation if we can't detect it easily from simple ASR wrapper
        # TODO: Enhance ASR wrapper to return detected language
        trans_source_lang = source_lang if source_lang != "auto" else "en"
        
        total_segments = len(segments)
        trans_start_time = time.time()
        
        for i, seg in enumerate(segments):
            if self.cancel_flag:
                raise InterruptedError("Task cancelled by user")
                
            # Progress update for translation
            elapsed = time.time() - trans_start_time
            avg_time = elapsed / max(i, 1) # Avoid div by zero initially
            remaining = avg_time * (total_segments - i) if i > 0 else 0
            etr_msg = f" (ETR: {format_remaining(remaining)})" if i > 0 else ""
            
            progress = 0.30 + (0.40 * (i / max(total_segments, 1))) # 30% to 70%
            yield ("progress", progress, f"Translating segment {i+1}/{total_segments}{etr_msg}...")
            
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
        yield ("progress", 0.70, "Generating subtitles...")
        src_srt_path = os.path.join(output_dir, "original.srt")
        trans_srt_path = os.path.join(output_dir, "translated.srt")
        
        self._generate_srt(segments, src_srt_path, text_key="text")
        self._generate_srt(dubbing_script, trans_srt_path, text_key="text")

        # 4. Generate Audio (TTS)
        yield ("progress", 0.75, "Generating dubbing audio...")
        if dubbing_enabled:
            print("Generating dubbing audio...")
            final_audio_path = os.path.join(output_dir, "final_dubbed.wav")
            
            tts_lang = self.lang_map.get(target_lang, "English")
            
            # Use generator for progress
            generator = self.dubber.generate_audio_track_iter(
                script=dubbing_script,
                output_path=final_audio_path,
                default_speaker=speaker,
                default_language=tts_lang,
                total_duration=video_duration
            )
            
            tts_start_time = time.time()
            
            for item in generator:
                if self.cancel_flag:
                    raise InterruptedError("Task cancelled by user")
                    
                if item[0] == "progress":
                    # ("progress", i, total, msg)
                    cur, total, msg = item[1], item[2], item[3]
                    
                    elapsed = time.time() - tts_start_time
                    avg_time = elapsed / max(cur, 1)
                    remaining = avg_time * (total - cur) if cur > 0 else 0
                    etr_msg = f" (ETR: {format_remaining(remaining)})" if cur > 0 else ""
                    
                    # Map TTS progress to 75% -> 90% range
                    overall_progress = 0.75 + (0.15 * (cur / max(total, 1)))
                    yield ("progress", overall_progress, f"{msg}{etr_msg}")
                    
                elif item[0] == "result":
                    # Final result path
                    pass
        else:
            print("Skipping Dubbing (using original audio)...")
            final_audio_path = audio_path # Use the extracted original audio
        
        # 5. Merge Video + Audio + Subtitles
        yield ("progress", 0.90, "Merging video, audio, and subtitles...")
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
                "-progress", "pipe:1", # Output progress to stdout
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
                "-progress", "pipe:1", # Output progress to stdout
                final_video_path
            ])
        
        print(f"Executing FFmpeg: {' '.join(cmd)}")
        
        # Use Popen to allow cancellation and progress reading
        # NOTE: -progress pipe:1 writes to stdout. FFmpeg logs go to stderr.
        # We redirect stderr to stdout to avoid buffer deadlock if stderr fills up.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8')
        
        ffmpeg_start_time = time.time()
        
        # We need to poll stdout non-blockingly or use threads.
        # Simple approach: Since -progress pipe:1 outputs small lines frequently, 
        # reading line-by-line is usually fine unless FFmpeg hangs without output.
        
        # Pattern for out_time_us=123456
        time_pattern = re.compile(r"out_time_us=(\d+)")
        
        # We can loop over proc.stdout
        if proc.stdout:
            for line in proc.stdout:
                if self.cancel_flag:
                    proc.terminate()
                    raise InterruptedError("Task cancelled by user")
                    
                line = line.strip()
                # Print FFmpeg log to console for debugging
                # print(f"[FFmpeg] {line}") 
                
                match = time_pattern.search(line)
                if match:
                    out_time_us = int(match.group(1))
                    current_sec = out_time_us / 1000000.0
                    
                    if video_duration > 0:
                        prog_ratio = min(current_sec / video_duration, 1.0)
                        
                        elapsed = time.time() - ffmpeg_start_time
                        if elapsed > 1 and prog_ratio > 0.01:
                            total_estimated = elapsed / prog_ratio
                            remaining = total_estimated - elapsed
                            etr_msg = f" (ETR: {format_remaining(remaining)})"
                        else:
                            etr_msg = ""
                            
                        # Map to 90-99%
                        ui_prog = 0.90 + (0.09 * prog_ratio)
                        yield ("progress", ui_prog, f"Rendering video (FFmpeg) {int(prog_ratio*100)}%{etr_msg}...")
        
        proc.wait()
            
        if proc.returncode != 0:
            # Stderr is merged into stdout, so we can't read it separately here.
            # But we could capture the last few lines of stdout if we were buffering it.
            # For now, just raise generic error.
            raise RuntimeError(f"FFmpeg failed with return code {proc.returncode}. Check console logs for details.")
        
        yield ("result", (final_audio_path, dubbing_script, src_srt_path, trans_srt_path, final_video_path))
