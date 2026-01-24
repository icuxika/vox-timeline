import whisper
import torch
from typing import List, Dict, Any, Optional

class WhisperASR:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper ASR model.
        
        Args:
            model_size: Size of the model (tiny, base, small, medium, large)
            device: Device to run on (cuda or cpu). If None, auto-detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file.
            language: Language code (e.g., 'en', 'zh'). If None, auto-detects.
            
        Returns:
            List of segments: [{'start': 0.0, 'end': 2.0, 'text': '...'}]
        """
        print(f"Transcribing {audio_path}...")
        
        # Transcribe
        result = self.model.transcribe(audio_path, language=language)
        
        return result["segments"]
