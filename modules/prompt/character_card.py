import re
try:
    import png
    PNG_PRESENT = True
except ModuleNotFoundError:
    PNG_PRESENT = False
import base64
import json
import yaml
from .chat_history import ChatMessage, ChatHistory

__all__ = ("CharacterCard",)


class CharacterCard:
    def __init__(self):
        self.name = str()
        self.description = str()
        self.scenario = str()
        self.greeting: ChatMessage
        self.example_messages: list[ChatHistory] = []

    def read_dict(self, dict: dict):
        self.name = dict["name"].strip()
        self.description = dict["description"].strip()
        self.scenario = dict.get("scenario", "").strip()
        self.greeting = ChatMessage(self.name, False, dict["first_mes"].strip())
        self.example_messages = []
        examples = filter(None, re.split("<start>", dict["mes_example"], flags=re.IGNORECASE))
        for example in examples:
            if not example.strip():
                continue
            history = ChatHistory(self)
            history.read(example.strip())
            self.example_messages.append(history)

    def load(self, file_path: str):
        if file_path.endswith('.json'):
            self._load_json(file_path)
        if file_path.endswith('.yaml'):
            self._load_yaml(file_path)
        elif file_path.endswith('.png'):
            self._load_img(file_path)
        elif "." not in file_path:
            for ext in ["json", "yaml", "png"]:
                try:
                    self.load(f"./characters/{file_path}.{ext}")
                    break
                except FileNotFoundError:
                    pass
            else:
                raise ValueError("File not found.")
        else:
            raise ValueError('Unsupported file format.')

    def _load_json(self, file_path: str):
        with open(file_path, "r") as file:
            self.read_dict(json.load(file))

    def _load_yaml(self, file_path: str):
        with open(file_path, "r") as file:
            self.read_dict(yaml.safe_load(file))

    def _load_img(self, file_path: str):
        if not PNG_PRESENT:
            raise Exception("Please install the pypng library to load image files.")

        # Get the chunks
        chunks = list(png.Reader(file_path).chunks())
        tEXtChunks = [chunk for chunkType, chunk in chunks if chunkType == b'tEXt']

        # Find the tEXt chunk containing the data
        data_chunk = None
        for tEXtChunk in tEXtChunks:
            if tEXtChunk.startswith(b'chara\x00'):
                data_chunk = tEXtChunk
                break

        if data_chunk is not None:
            # Extract the data from the tEXt chunk
            base64EncodedData = data_chunk[6:].decode('utf-8')
            data = base64.b64decode(base64EncodedData).decode('utf-8')

            return self.read_dict(json.loads(data))
        else:
            return None

    def save_json(self, file_path: str):
        json_data = {
            "name": self.name,
            "description": self.description,
            "scenario": self.scenario,
            "first_mes": self.greeting.message,
            "mes_example": "<start>\n" + "<start>\n".join(log.to_string_log() for log in self.example_messages)
        }
        with open(file_path, "w") as file:
            json.dump(json_data, file)