from modules.prompt import Prompt

__all__ = ("TextPrompt",)


class TextPrompt(Prompt):
    def __init__(self):
        self.text = ""

    def init(self, text: str):
        self.text = text

    def get_stop_sequences(self) -> list[str]:
        return []

    def to_string(self) -> str:
        return self.text