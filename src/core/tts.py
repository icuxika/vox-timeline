import torch
import soundfile as sf
import os
from qwen_tts import Qwen3TTSModel
from typing import Optional, Tuple, List
import numpy as np

class TTSEngine:
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device: str = "cuda:0"):
        print(f"Loading TTS model: {model_name} on {device}...")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        print("TTS model loaded.")

    def generate(self, 
                 text: str, 
                 speaker: str = "Uncle_Fu", 
                 language: str = "Chinese", 
                 instruct: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text.
        Returns: (audio_waveform, sample_rate)
        """
        # Ensure text is not empty
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct
        )
        
        # wavs[0] is the audio data (numpy array or tensor)
        # Convert to numpy if it's a tensor
        audio_data = wavs[0]
        if isinstance(audio_data, torch.Tensor):
            # Qwen3-TTS uses bfloat16 internally, we must convert to float32 for audio libraries
            audio_data = audio_data.to(torch.float32).cpu().numpy()
            
        return audio_data, sr
