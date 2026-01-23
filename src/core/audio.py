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

    def export(self, output_path: str, format: str = "wav"):
        """
        Export the composite audio to a file.
        """
        if not self.segments:
            print("No segments to export.")
            return

        # 1. Determine total duration
        max_duration_ms = 0
        for start_ms, seg in self.segments:
            end_ms = start_ms + len(seg)
            if end_ms > max_duration_ms:
                max_duration_ms = end_ms

        # 2. Create silent base track
        # Pydub silence is usually created with .silent()
        # Ensure we have at least some length
        if max_duration_ms == 0:
            max_duration_ms = 1000 # 1 sec min
            
        final_audio = AudioSegment.silent(duration=max_duration_ms)

        # 3. Overlay all segments
        print(f"Composing audio (duration: {max_duration_ms/1000:.2f}s)...")
        for start_ms, seg in self.segments:
            final_audio = final_audio.overlay(seg, position=start_ms)

        # 4. Export
        final_audio.export(output_path, format=format)
        print(f"Exported audio to {output_path}")
