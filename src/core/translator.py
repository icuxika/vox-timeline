import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import List, Dict, Optional

class TranslateGemma:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TranslateGemma, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_id: str = "google/translategemma-4b-it"):
        if self.initialized:
            return
            
        self.model_id = model_id
        print(f"Loading Translation model '{model_id}'...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
            self.initialized = True
        except Exception as e:
            print(f"Error loading TranslateGemma: {e}")
            raise

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using TranslateGemma.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (e.g., 'en', 'zh').
            target_lang: Target language code (e.g., 'en', 'zh').
            
        Returns:
            Translated text.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = len(inputs['input_ids'][0])
        
        # Generate
        with torch.inference_mode():
            generation = self.model.generate(**inputs, do_sample=True)
            
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
