from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

__all__ = ("TFModel",)


class TFModel(LanguageModel):
    def __init__(self, model_path: str, max_context: int, lora_path: str = None, auxiliary: bool = False):
        assert(isinstance(model_path, str))
        if not model_path:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model path using the argument --model.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model path using the argument --auxiliary-model.")
            exit(-1)
        self.new_seed()
        
        self.model = LlamaForCausalLM.from_pretrained(model_path,
                                                     #cache_dir="./s_cache",
                                                     load_in_8bit=True,
                                                     device_map={"": 0})
                                                     #use_flash_attention_2=True)
        self.model.config.use_cache = True

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"

        self.peftModel = None
        if lora_path:
            self.peftModel = PeftModel.from_pretrained(self.model, lora_path)
            self.peftModel.config.use_cache = False

        self.pipe = pipeline(task="text-generation", model=self.peftModel if self.peftModel else self.model, tokenizer=self.tokenizer)
        
        super().__init__(max_context, auxiliary)

    def __del__(self):
        del self.model
        del self.tokenizer
        del self.peftModel

    def wait(self):
        pass

    @override
    def _generate_once(self, data: dict) -> str:
        raise NotImplementedError()
    
    @override
    def supports_batching(self) -> bool:
        return True
    
    @override
    def generate_batch(self, prompts: list[str], batch_size: int = 1, max_tokens: int = 8) -> list[str]:
        generate_kwargs = {
            "do_sample": False,
            "temperature": 1,
            "repetition_penalty": 1,
            "max_new_tokens": max_tokens,
        }
        results = self.pipe(prompts, batch_size=batch_size, **generate_kwargs)
        results = [o[0]['generated_text'][len(p):] for p, o in zip(prompts, results)]
        return results