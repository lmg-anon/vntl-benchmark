from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore

from transformers import pipeline
from unsloth import FastLanguageModel

__all__ = ("UnslothModel",)


class UnslothModel(LanguageModel):
    def __init__(self, model_path: str, max_context: int, lora_path: str = None, auxiliary: bool = False):
        assert(isinstance(model_path, str))
        if not model_path:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model path using the argument --model.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model path using the argument --auxiliary-model.")
            exit(-1)
        self.new_seed()

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = lora_path if lora_path else model_path,
            cache_dir = "./s_cache",
            max_seq_length = max_context,
            dtype = None,
            load_in_4bit = False,
        )
        self.tokenizer.padding_side = "left"

        self.pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer)
        
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