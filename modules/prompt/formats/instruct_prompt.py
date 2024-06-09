from modules.prompt import Prompt, ChatHistory, ChatMessage
from dataclasses import dataclass
import yaml
import pybars
import html

__all__ = ("InstructConfig", "InstructPrompt",)


@dataclass(frozen=True)
class InstructConfig:
    system: str = ""
    guidance: str = ""
    first_input_sequence_start: str | None =  None
    input_sequence_start: str = ""
    input_sequence_end: str = ""
    first_output_sequence_start: str | None =  None
    output_sequence_start: str = ""
    output_sequence_end: str = ""
    newline_wrap: bool = False
    include_names: bool = False
    newline_merge: bool = False
    trim: bool = True

    @classmethod
    def from_file(cls, file_path: str) -> 'InstructConfig':
        with open(file_path, "r") as f:
            data = yaml.safe_load(f.read())
        return cls.from_data(data.get("instruct", {}))

    @classmethod
    def from_data(cls, data: dict) -> 'InstructConfig':
        return cls(**data)

class InstructPrompt(Prompt):
    def __init__(self, config: InstructConfig):
        self.config = config
        self.history = ChatHistory()
        self.stop_sequences: list[str] = []

    def init(self):
        self.history.clear()
        self.stop_sequences.clear()

    def add_instruction(self, text):
        self.history.add("", text, True)

    def add_response(self, text):
        self.history.add("", text, False)

    def add_stop_sequence(self, sequence: str):
        self.stop_sequences.append(sequence)

    def add_stop_sequences(self, sequences: list[str]):
        self.stop_sequences += sequences

    def get_stop_sequences(self) -> list[str]:
        return [self.config.input_sequence_start, *self.stop_sequences]

    def to_string(self) -> str:
        def format_chat_history(msg: ChatMessage, first: bool, last: bool, is_example: bool):
            sequence_start = self.config.input_sequence_start if msg.is_user else self.config.output_sequence_start
            sequence_end = self.config.input_sequence_end if msg.is_user else self.config.output_sequence_end
            if last and not is_example:
                sequence_end = ""
            return f"{sequence_start}{msg.message}{sequence_end}"

        compiler = pybars.Compiler()

        macros = {
            "guidance": self.config.guidance
        }
        macros["guidance"] = html.unescape(compiler.compile(self.config.guidance)(macros)).strip()

        value: str = html.unescape(compiler.compile(self.config.system)(macros))
        if len(self.history) > 0 and self.history[-1].is_user:
            self.history.add("", "", False)
        value += self.history.to_string(format_chat_history)
        return value