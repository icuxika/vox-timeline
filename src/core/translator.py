import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional

class HymtTranslator:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HymtTranslator, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, model_id: str = "tencent/HY-MT1.5-1.8B"):
        if self.initialized:
            return
            
        self.model_id = model_id
        print(f"Loading Translation model '{model_id}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            self.initialized = True
            
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
        except Exception as e:
            print(f"Error loading HymtTranslator: {e}")
            raise

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # Convert code to full name for prompt
        target_lang_name = self.lang_map.get(target_lang, target_lang)
        
        messages = [
            {"role": "user", "content": f"Translate the following segment into {target_lang_name}, without additional explanation.\n\n{text}"},
        ]
        
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        input_ids = tokenized_chat.to(self.model.device)
        input_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=2048)
            
        # Decode only the new tokens
        output_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return output_text.strip()

class HelsinkiOpusTranslator:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HelsinkiOpusTranslator, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.models = {}
        self.tokenizers = {}
        self.initialized = True
        
    def _get_model_pair(self, source_lang: str, target_lang: str):
        # Handle code mapping if necessary (e.g. zh -> zh_CN in some older models, but opus-mt usually uses generic codes)
        # opus-mt-en-zh exists.
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        if model_name not in self.models:
            print(f"Loading Translation model '{model_name}'...")
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                raise RuntimeError(f"Translation model for {source_lang}->{target_lang} not available or failed to load.")
                
        return self.models[model_name], self.tokenizers[model_name]

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        model, tokenizer = self._get_model_pair(source_lang, target_lang)
        
        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            translated = model.generate(**inputs)
            
        return tokenizer.decode(translated[0], skip_special_tokens=True)

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
