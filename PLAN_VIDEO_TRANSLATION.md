# Video Translation & Dubbing Plan

## 1. Objective
Implement an end-to-end pipeline to:
1. Extract audio from a video file.
2. Transcribe the audio using OpenAI Whisper (ASR) to get text and timestamps.
3. Translate the transcribed text to a target language (e.g., English -> Chinese).
4. Generate dubbing audio for the translated text using the existing `VideoDubber`.
5. (Optional) Merge the new audio back into the video.

## 2. Architecture

### New Modules
- **`src/core/asr.py`**: Wraps `openai-whisper`.
  - `WhisperASR`: Loads model, handles transcription, returns segments with `start`, `end`, `text`.
- **`src/core/translator.py`**: Handles text translation.
  - `TextTranslator`: Uses `deep-translator` (or similar) to translate text segments.
- **`src/pipeline/translator_pipeline.py`**: Orchestrates the flow.
  - `VideoTranslator`: 
    - Input: Video path, Source Lang (optional), Target Lang.
    - Process: Video -> Audio -> ASR -> Script -> Translation -> Translated Script -> VideoDubber -> Audio Track.

### Existing Component Updates
- **`app.py`**: Add a new tab "Video Dubbing (Auto-Translate)".
  - Inputs: Video File, Model Size (tiny/base/small/medium/large), Target Language, Speaker.
  - Outputs: Translated Script (editable), Generated Audio, Final Video (optional).

## 3. Data Flow
1. **Input**: `movie.mp4`
2. **Extraction**: `temp_audio.wav` (via MoviePy/FFmpeg)
3. **ASR**: Whisper -> `[{"start": 0.0, "end": 2.0, "text": "Hello"}]`
4. **Translation**: -> `[{"start": 0.0, "end": 2.0, "text": "你好", "instruct": "neutral"}]`
5. **Dubbing**: `VideoDubber` -> `final_dubbing.wav`

## 4. Key Challenges & Solutions
- **Timing Mismatch**: Translated text duration may not match original.
  - *Solution*: Use the original `start` time. Rely on `VideoDubber`'s auto-shift (already implemented) to prevent overlap if the new audio is longer.
- **Translation Context**: Sentence-by-sentence translation might lose context.
  - *Solution*: Start with simple sentence-level translation. Future upgrade: Context-aware translation (LLM).

## 5. Dependencies
- `openai-whisper` (Existing)
- `deep-translator` (New, for translation)
- `moviepy` (Existing, for media handling)

## 6. Implementation Steps
1. Add `deep-translator` to `pyproject.toml`.
2. Implement `src/core/asr.py`.
3. Implement `src/core/translator.py`.
4. Create `src/pipeline/translator_pipeline.py`.
5. Update `app.py` UI.
