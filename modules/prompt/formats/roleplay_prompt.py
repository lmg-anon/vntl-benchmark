from modules.prompt import CharacterCard, ChatHistory, ChatMessage, Prompt
from modules.prompt.formats import InstructConfig
from dataclasses import dataclass
import pybars
import html
import yaml

__all__ = ("RoleplayConfig", "RoleplayPrompt",)


@dataclass(frozen=True)
class RoleplayConfig(InstructConfig):
    example_separator: str = ""
    start: str = ""

    # # Override Instruct properties if explicitly provided in 'roleplay' section
    # def __post_init__(self):
    #     for key in self.__dict__:
    #         if self.__dict__[key] in [None, ""]:
    #             object.__setattr__(self, key, getattr(InstructConfig, key, ""))

    @classmethod
    def from_file(cls, file_path: str) -> 'RoleplayConfig':
        with open(file_path, "r") as f:
            data = yaml.safe_load(f.read())
        base = InstructConfig.from_data(data.get("instruct", {}))
        combined_data = {**base.__dict__, **data.get("roleplay", {})}
        return cls.from_data(combined_data)

    @classmethod
    def from_data(cls, data: dict) -> 'RoleplayConfig':
        return cls(**data)

class RoleplayPrompt(Prompt):
    def __init__(self, config: RoleplayConfig):
        self.config = config
        self.user_name = str()
        self.card: CharacterCard
        self.history: ChatHistory
        self.stop_sequences: list[str] = []

    def init(self, user_name: str, card: CharacterCard, add_greeting: bool = True):
        self.user_name = user_name
        self.card = card
        self.history = ChatHistory(card)
        if add_greeting:
            self.add_message(self.card.greeting.sender, self.card.greeting.message)
        self.stop_sequences.clear()

    def add_messages_from_file(self, file_path: str):
        history = ChatHistory(self.card)
        history.load(file_path)
        self.add_messages_from(history)

    def add_messages_from(self, other: ChatHistory):
        self.history.add_from(other)

    def add_message(self, sender: str, msg: str, is_user: bool | None = None):
        self.history.add(sender, msg, is_user)

    def pop_message(self) -> ChatMessage:
        return self.history.pop()

    def add_stop_sequence(self, sequence: str):
        self.stop_sequences.append(sequence)

    def add_stop_sequences(self, sequences: list[str]):
        self.stop_sequences += sequences

    def get_stop_sequences(self) -> list[str]:
        sequences = [self.config.input_sequence_start, *self.stop_sequences]
        sequences = [s.replace("{{char}}", self.card.name).replace("<BOT>", self.card.name) for s in sequences]
        sequences = [s.replace("{{user}}", self.user_name).replace("<USER>", self.user_name) for s in sequences]
        return sequences

    def to_string(self) -> str:
        def format_chat_history(msg: ChatMessage, first: bool, last: bool, is_example: bool):
            sequence_start = self.config.input_sequence_start if msg.is_user else self.config.output_sequence_start
            if first:
                if msg.is_user and self.config.first_input_sequence_start is not None:
                    sequence_start = self.config.first_input_sequence_start
                elif not msg.is_user and self.config.first_output_sequence_start is not None:
                    sequence_start = self.config.first_output_sequence_start
            sequence_end = self.config.input_sequence_end if msg.is_user else self.config.output_sequence_end
            if last and not is_example:
                sequence_end = ""
            message = f"{msg.sender}: {msg.message}" if self.config.include_names else msg.message
            return f"{sequence_start}{message}{sequence_end}"

        compiler = pybars.Compiler()

        macros = {
            "guidance": self.config.guidance,
            "char": self.card.name,
            "user": self.user_name,
            "description": self.card.description,
            "scenario": self.card.scenario
        }
        macros["guidance"] = html.unescape(compiler.compile(self.config.guidance)(macros)).strip()

        value: str = html.unescape(compiler.compile(self.config.system)(macros))

        if self.card.example_messages:
            value += self.config.example_separator
            value += self.config.example_separator.join([log.to_string(format_chat_history, True) for log in self.card.example_messages])

        if self.config.start:
            value += self.config.start + "\n"

        #value += self.instruct_format["separator_sequence"]
        value += self.history.to_string(format_chat_history)
        value = value.replace("\r\n", "\n")
        value = value.replace("{{char}}", self.card.name).replace("<BOT>", self.card.name)
        value = value.replace("{{user}}", self.user_name).replace("<USER>", self.user_name)

        if self.config.newline_merge:
            # Merge consecutive linebreaks into one to mimic the simple-proxy behavior
            replaced_value = value.replace("\n\n", "\n")
            while replaced_value != value:
                value = replaced_value
                replaced_value = value.replace("\n\n", "\n")

        if self.config.trim:
            value = value.strip()

        return value