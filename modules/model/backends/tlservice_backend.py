from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore
import time
import html

import importlib
if importlib.util.find_spec("translators") is None:
    raise ModuleNotFoundError()

__all__ = ("TLServiceModel",)


class TLServiceModel(LanguageModel):
    def __init__(self, translator: str, auxiliary: bool = False):
        import translators as ts
        #_ = ts.preaccelerate_and_speedtest()
        self.model = ts
        self.translator = translator
        super().__init__(1024, auxiliary)

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        time.sleep(0.432)
        return html.unescape(self.model.translate_text(data["prompt"], from_language='ja', to_language='en', translator=self.translator))
    
    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8, stop_sequences: list[str] = []) -> list[str]:
        raise NotImplementedError()