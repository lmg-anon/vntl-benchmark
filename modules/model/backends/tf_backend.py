from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore

import importlib
if importlib.util.find_spec("transformers") is None:
    raise ModuleNotFoundError()

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

        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

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

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def supports_batching(self) -> bool:
        return True
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8, stop_sequences: list[str] = []) -> list[str]:
        assert not stop_sequences
        generate_kwargs = {
            "do_sample": False,
            "temperature": 1,
            "repetition_penalty": 1,
            "max_new_tokens": max_tokens,
        }
        results = self.pipe(prompts, batch_size=len(prompts), **generate_kwargs)
        results = [o[0]['generated_text'][len(p):] for p, o in zip(prompts, results)]
        return results