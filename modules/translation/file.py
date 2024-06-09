__all__ = ("TranslationFile", "TranslationEntry",)


class TranslationEntry:
    def __init__(self, japanese: str, english: str, fidelity: str = None, id: str = None):
        self.japanese = japanese.strip()
        self.english = english.strip()
        self.fidelity = fidelity
        self.book_id = id

    def __str__(self):
        return f"Japanese: {self.japanese}\n" \
               f"English: {self.english}\n" \
               f"Fidelity: {self.fidelity}\n" \
               f"BookID: {self.book_id}\n"

class TranslationFile:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entries: list[TranslationEntry] = []

    def read_file(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_id = None
        current_japanese = None
        current_english = None
        current_fidelity = None

        for line in lines:
            line = line.strip()
            if line.startswith('<<JAPANESE>>'):
                # If there's a previous entry, commit it
                if current_japanese is not None and current_english is not None:
                    entry = TranslationEntry(current_japanese, current_english, current_fidelity, current_id)
                    self.entries.append(entry)
                current_japanese = ''
                current_english = current_fidelity = None
            elif line.startswith('<<ENGLISH>>'):
                current_fidelity = self.parse_fidelity(line)
                current_english = ''
            elif line.startswith('<<ID>>'):
                current_id = self.parse_id(line)
            elif line == '<<NEW_FILE>>':
                # If there's a previous entry, commit it
                if current_japanese is not None and current_english is not None:
                    entry = TranslationEntry(current_japanese, current_english, current_fidelity, current_id)
                    self.entries.append(entry)
                # Reset for a new file/entry
                current_id = current_japanese = current_english = current_fidelity = None
            elif line.startswith('<<'):
                assert False, line
            elif current_japanese is not None and current_english is None:
                current_japanese += line + '\n'
            elif current_english is not None:
                current_english += line + '\n'

        # Add the last entry if the file doesn't end with <<NEW_FILE>>
        if current_japanese is not None and current_english is not None:
            entry = TranslationEntry(current_japanese.strip(), current_english.strip(), current_fidelity, current_id)
            self.entries.append(entry)

    @staticmethod
    def parse_fidelity(line: str):
        if '(fidelity' in line:
            start = line.find('(')
            end = line.find(')')
            return line[start+1:end]
        return None

    @staticmethod
    def parse_id(line: str):
        if '<<ID>>' in line:
            start = line.find('<<ID>>')
            end = line.find('<</ID>>')
            return line[start+6:end]
        return None