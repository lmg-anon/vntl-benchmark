from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore

import importlib
if importlib.util.find_spec("fairseq") is None:
    raise ModuleNotFoundError()

import logging
logging.disable(logging.CRITICAL)

__all__ = ("SugoiModel",)


class SugoiModel(LanguageModel):
    def __init__(self, model_path: str, auxiliary: bool = False):
        assert(isinstance(model_path, str))
        if not model_path:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model path using the argument --model.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model path using the argument --auxiliary-model.")
            exit(-1)
        
        from fairseq.models.transformer import TransformerModel
        self.model = TransformerModel.from_pretrained(
            'G:/JA2EN/fairseq/japaneseModel/',
            checkpoint_file='big.pretrain.pt',
            source_lang="ja",
            target_lang="en",
            bpe='sentencepiece',
            sentencepiece_model='G:/JA2EN/fairseq/spmModels/spm.ja.nopretok.model',
            is_gpu=True
        ).cuda()
        
        super().__init__(1024, auxiliary)

    def __del__(self):
        del self.model

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        return self.model.translate([data["prompt"]])[0]
    
    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def supports_batching(self) -> bool:
        return True
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8, stop_sequences: list[str] = []) -> list[str]:
        return self.model.translate(prompts)