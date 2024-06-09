from typing import TYPE_CHECKING, Any, Callable
if TYPE_CHECKING:
    from . import CharacterCard
import json
from dataclasses import dataclass

__all__ = ("ChatMessage", "ChatHistory",)


@dataclass
class ChatMessage:
    sender: str
    is_user: bool
    message: str

    def __repr__(self) -> str:
        return f"{self.sender}: {self.message}" if self.sender else self.message

    def to_string(self, formatter: Callable, first: bool, last: bool, is_example: bool) -> str:
        return formatter(self, first, last, is_example)

class ChatHistory:
    def __init__(self, character: 'CharacterCard | None' = None):
        self.character = character
        self.entries: list[ChatMessage] = []

    def __repr__(self) -> str:
        return repr("\n".join([str(entry) for entry in self.entries]))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: Any) -> ChatMessage:
        return self.entries[index]

    def read(self, text: str):
        self.entries = []

        sender = None
        message = ""
        for line in text.splitlines():
            if ":" in line:
                if sender is not None:
                    is_user = (self.character is None or sender != self.character.name) and sender != "{{char}}"
                    self.entries.append(ChatMessage(sender, is_user, message.strip()))

                sender, message = [s.strip() for s in line.split(':', 1)]
            else:
                message += " " + line.strip()

        if sender is not None:
            is_user = (self.character is None or sender != self.character.name) and sender != "{{char}}"
            self.entries.append(ChatMessage(sender, is_user, message.strip()))

    def load(self, file_path: str):
        with open(file_path, 'r') as file:
            if file_path.endswith('.jsonl'):
                self._load_jsonl(file)
            elif file_path.endswith('.txt'):
                self.read(file.read())
            else:
                raise ValueError('Unsupported file format.')

    def _load_jsonl(self, file):
        for line in file:
            entry = json.loads(line)
            if 'name' not in entry:
                continue
            self.entries.append(ChatMessage(entry['name'], entry['is_user'], entry['mes']))

    def add(self, sender: str, message: str, is_user: bool | None = None):
        self.entries.append(ChatMessage(sender, is_user if is_user is not None else (self.character is None or sender != self.character.name) and sender != "{{char}}", message))

    def add_from(self, other: 'ChatHistory'):
        self.entries.extend(other.entries)

    def pop(self) -> ChatMessage:
        return self.entries.pop()

    def clear(self):
        self.entries.clear()

    def to_string(self, formatter: Callable, is_example: bool = False) -> str:
        result = str()
        for idx, entry in enumerate(self.entries):
            result += entry.to_string(formatter, idx == 0, idx == len(self.entries) - 1, is_example)
        return result

    def to_string_log(self) -> str:
        return "\n".join([f"{entry.sender}: {entry.message}" for entry in self.entries])