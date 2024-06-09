import abc

__all__ = ("Prompt",)


class Prompt(abc.ABC):
    @abc.abstractmethod
    def get_stop_sequences(self) -> list[str]:
        return []

    @abc.abstractmethod
    def to_string(self) -> str:
        return ""