from .character import Character

__all__ = ("VisualNovel",)


class VisualNovel:
    def __init__(self, title: str):
        self.title = title
        self.characters: list[Character] = []

    def get_character(self, name):
        for character in self.characters:
            if name == character.fp_name:
                return character
            elif name == character.japanese_fp_name:
                return character
        return None