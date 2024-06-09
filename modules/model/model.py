from typing import Iterator
from modules.log import Logger
from modules.prompt import Prompt
from modules.prompt.formats import *
import json
import random
import abc
import re

__all__ = ("LanguageModel",)


class LanguageModel(abc.ABC):
    base_seed = None

    def __init__(self, max_context: int, auxiliary: bool):
        self.max_context = max_context
        self.presets = dict()
        self.seed = LanguageModel.base_seed
        self.is_auxiliary = auxiliary

    def load_preset(self, file_path: str):
        with open(file_path, "r") as file:
            self.presets = json.load(file)

    def new_seed(self):
        self.seed = random.randint(1, 0xFFFFFFFF)
        Logger.log(f"New {'auxiliary ' if self.is_auxiliary else ''}model seed: {self.seed}", True)

    def clear_seed(self):
        self.seed = LanguageModel.base_seed

    def get_identifier(self) -> str:
        return f"Model backend{' (auxiliary)' if self.is_auxiliary else ''}"

    @abc.abstractmethod
    def wait(self):
        pass

    @abc.abstractmethod
    def _abort(self):
        pass

    @abc.abstractmethod
    def _generate_once(self, data: dict) -> Iterator[str]:
        return ""

    def generate_iter(self, prompt: Prompt | str, max_tokens: int = 8, stop_sequences: list[str] = []) -> Iterator[tuple[str, str]]:
        stop_sequences = stop_sequences if stop_sequences else (prompt.get_stop_sequences() if not isinstance(prompt, str) else [])
        data = self.presets.copy()
        data.update({
            "max_context_length": self.max_context,
            "max_length": max_tokens,
            "stop_sequence": stop_sequences,
            "stream": True,
            "prompt": prompt.to_string() if not isinstance(prompt, str) else prompt
        })

        if self.seed:
            data["sampler_seed"] = self.seed

        output_str = str()
        for iter, response_text in enumerate(self._generate_once(data)):
            # Skip if we couldn't generate anything.
            if not response_text:
                continue

            output_str += response_text
            #Logger.write(f"Gen {iter}: {repr(response_text)}", True)
            if any((match := s) in output_str for s in stop_sequences):
                yield response_text.split(match, 2)[0], output_str.split(match, 2)[0]
                break
            yield response_text, output_str

        self._abort()

    def generate(self, prompt: Prompt | str, max_tokens: int = 8, stop_sequences: list[str] = []) -> str:
        result = str()
        for _, output in self.generate_iter(prompt, max_tokens, stop_sequences):
            result = output
        return result

    def generate_until(self, prompt: Prompt | str, success: list[str] = [], failure: list[str] = [], ignore_case: bool = True, default: bool = False, max_tokens: int = 8, stop_sequences: list[str] = []) -> tuple[bool, str]:
        def remove_repeated_chars(str: str) -> str:
            return re.sub(r'([^\d\W])\1+', r'\1', str)
        
        if not success and not failure:
            raise Exception("No condition set, please use the generate() method instead.")

        success = [remove_repeated_chars(s.lower() if ignore_case else s) for s in success]
        failure = [remove_repeated_chars(s.lower() if ignore_case else s) for s in failure]

        result, result_gen = default, ""
        for _, output in self.generate_iter(prompt, max_tokens, stop_sequences):
            result_gen = output
            output = remove_repeated_chars(output)
            if any(word in (output.lower() if ignore_case else output) for word in success):
                result = True
                break
            if any(word in (output.lower() if ignore_case else output) for word in failure):
                result = False
                break

        return result, result_gen
    
    def supports_batching(self) -> bool:
        return False
    
    @abc.abstractmethod
    def generate_batch(self, prompts: list[str], batch_size: int = 1, max_tokens: int = 8) -> list[str]:
        return []