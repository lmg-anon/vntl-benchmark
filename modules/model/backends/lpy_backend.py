from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore

import importlib
if importlib.util.find_spec("llama_cpp") is None:
    raise ModuleNotFoundError()


__all__ = ("LpyModel",)


class LpyModel(LanguageModel):
    def __init__(self, model_path: str, max_context: int, auxiliary: bool = False):
        assert(isinstance(model_path, str))
        if not model_path:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model path using the argument --model.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model path using the argument --auxiliary-model.")
            exit(-1)
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=max_context, seed=self.seed, verbose=False)  # type: ignore
        super().__init__(max_context, auxiliary)

    def __del__(self):
        del self.llm

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        output = self.llm(
            data["prompt"],
            max_tokens=data["max_length"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            echo=False,
            stop=data["stop_sequence"],
            #frequency_penalty=data["frequency_penalty"],
            #presence_penalty=data["presence_penalty"],
            repeat_penalty=data["rep_pen"],
            top_k=data["top_k"],
            #stream=True,
            tfs_z=data["tfs"]
        )
        return output["choices"][0]["text"]  # type: ignore
    
    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8) -> list[str]:
        raise NotImplementedError()