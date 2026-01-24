from pydub import AudioSegment
import numpy as np
import io
import soundfile as sf

class AudioTimeline:
    def __init__(self):
        self.segments = [] # List of (start_ms, AudioSegment)

    def add_segment(self, start_time: float, audio_data: np.ndarray, sample_rate: int, auto_shift: bool = True):
        """
        Add an audio segment to the timeline.
        If auto_shift is True, it will check if the previous segment overlaps with this one,
        and if so, shift the start_time of this segment to be after the previous one ends.
        """
        # Convert numpy array to AudioSegment
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
            wav_buffer.seek(0)
            segment = AudioSegment.from_wav(wav_buffer)

        start_ms = int(start_time * 1000)

        if auto_shift and self.segments:
            # Check overlap with the last added segment
            # Assuming segments are added in chronological order of their INTENDED start times
            last_start_ms, last_seg = self.segments[-1]
            last_end_ms = last_start_ms + len(last_seg)
            
            # Add a small buffer (e.g. 50ms) to avoid too tight splicing
            min_start_ms = last_end_ms + 50 
            
            if start_ms < min_start_ms:
                print(f"  [Auto-Shift] Segment overlap detected. Shifted start from {start_ms}ms to {min_start_ms}ms (+{min_start_ms - start_ms}ms)")
                start_ms = min_start_ms

        self.segments.append((start_ms, segment))

    def export(self, output_path: str, format: str = "wav", target_duration_ms: int = None):
        """
        Export the composite audio to a file.
        
        Args:
            output_path: Path to save the audio file.
            format: Audio format (default: wav).
            target_duration_ms: If provided, force the output audio to be exactly this duration (in ms).
                                If content is shorter, pad with silence.
                                If content is longer, trim the end.
        """
        if not self.segments:
            print("No segments to export.")
            # If target duration is set, we should export a silent file of that length
            if target_duration_ms:
                print(f"Generating silent audio of {target_duration_ms}ms...")
                final_audio = AudioSegment.silent(duration=target_duration_ms)
                final_audio.export(output_path, format=format)
                print(f"Exported silent audio to {output_path}")
            return

        # 1. Determine total duration (based on content)
        content_duration_ms = 0
        for start_ms, seg in self.segments:
            end_ms = start_ms + len(seg)
            if end_ms > content_duration_ms:
                content_duration_ms = end_ms

        # 2. Determine final duration
        final_duration_ms = content_duration_ms
        if target_duration_ms is not None:
            final_duration_ms = target_duration_ms
            
        if final_duration_ms == 0:
            final_duration_ms = 1000 # 1 sec min safety

        # 3. Create silent base track
        final_audio = AudioSegment.silent(duration=final_duration_ms)

        # 4. Overlay all segments
        print(f"Composing audio (duration: {final_duration_ms/1000:.2f}s)...")
        for start_ms, seg in self.segments:
            # Skip segments that start after the target duration
            if start_ms >= final_duration_ms:
                print(f"  [Warning] Segment at {start_ms}ms skipped (beyond target duration {final_duration_ms}ms)")
                continue
                
            # If segment extends beyond target, it will be naturally cropped by the base track length 
            # ONLY IF we don't expand. Pydub's overlay usually expands if the overlay is longer.
            # So we should crop the segment if needed.
            if start_ms + len(seg) > final_duration_ms:
                 remaining_space = final_duration_ms - start_ms
                 if remaining_space > 0:
                     seg = seg[:remaining_space]
                 else:
                     continue

            final_audio = final_audio.overlay(seg, position=start_ms)

        # Ensure exact length (in case overlay expanded it, though we tried to prevent it)
        if len(final_audio) != final_duration_ms:
            final_audio = final_audio[:final_duration_ms]
            
        # 5. Export
        final_audio.export(output_path, format=format)
        print(f"Exported audio to {output_path}")
