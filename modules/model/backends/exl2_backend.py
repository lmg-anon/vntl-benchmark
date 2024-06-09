from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

__all__ = ("EXL2Model",)


class EXL2Model(LanguageModel):
    def __init__(self, model_path: str, max_context: int, max_batch_size: int = 1, lora_path: str = None, auxiliary: bool = False):
        assert(isinstance(model_path, str))
        if not model_path:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model path using the argument --model.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model path using the argument --auxiliary-model.")
            exit(-1)
        self.new_seed()

        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()

        config.max_batch_size = max_batch_size  # Model instance needs to allocate temp buffers to fit the max batch size
        config.max_seq_len = max_context
        config.max_input_len = min(config.max_seq_len, 2048)
        config.max_attn_size = min(config.max_seq_len, 2048)**2
        config.no_flash_attn = True

        self.model = ExLlamaV2(config)
        
        cache = ExLlamaV2Cache(self.model, lazy = True, batch_size = max_batch_size)  # Cache needs to accommodate the batch size
        self.model.load_autosplit(cache)

        self.tokenizer = ExLlamaV2Tokenizer(config)

        # Initialize generator
        self.generator = ExLlamaV2BaseGenerator(self.model, cache, self.tokenizer)
        
        super().__init__(max_context, auxiliary)

    def __del__(self):
        del self.model
        del self.tokenizer
        del self.generator

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
        # Sampling settings
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.01
        settings.top_k = 1 # Same as temperature 0
        settings.top_p = 0
        settings.typical = 0
        settings.token_repetition_penalty = 1

        results = self.generator.generate_simple(prompts, settings, max_tokens, seed = self.seed)
        results = [o[len(p):] for p, o in zip(prompts, results)]
        return results