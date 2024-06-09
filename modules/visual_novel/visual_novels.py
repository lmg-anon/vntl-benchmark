from .visual_novel import VisualNovel
from .character import Character

__all__ = ("VisualNovels",)


class VisualNovels:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entries: list[VisualNovel] = []

    def read_file(self):
        current_book = None
        current_character = None
        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    # Start of a new book
                    if current_book:
                        if current_character:
                            current_book.characters.append(current_character)
                            current_character = None
                        self.entries.append(current_book)
                    current_book = VisualNovel(title=line[1:-1])
                elif line.startswith('Name:'):
                    # Start of a new character
                    if current_character:
                        current_book.characters.append(current_character)
                    name = line.split(':', 1)[1].strip()
                    current_character = Character(name, None, None)
                elif line.startswith('Gender:'):
                    current_character.gender = line.split(':', 1)[1].strip()
                elif line.startswith('Aliases:'):
                    aliases = line.split(':', 1)[1].strip()
                    current_character.aliases = aliases if aliases != 'None' else None
                else:
                    assert not line
            if current_character:
                current_book.characters.append(current_character)
            if current_book:
                self.entries.append(current_book)

    def get_character(self, vn_title, char_name) -> Character | None:
        for book in self.entries:
            if book.title != vn_title:
                continue
            if char := book.get_character(char_name):
                return char
        return None