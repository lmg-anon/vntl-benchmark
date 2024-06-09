from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore
import requests
import json
import time
import sseclient
import math

__all__ = ("OaiModel",)


class OaiModel(LanguageModel):
    def __init__(self, oai_host: str, api_key: str, max_context: int, auxiliary: bool = False):
        assert(isinstance(oai_host, str))
        if not oai_host:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model backend host using the argument --host.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model backend host using the argument --auxiliary-host.")
            exit(-1)
        self.oai_host = oai_host.strip('/')
        self.api_key = api_key
        if not self.oai_host.startswith("http"):
            self.oai_host = f"http://{self.oai_host}"
        super().__init__(max_context, auxiliary)

    def wait(self):
        wait_started = False
        while True:
            try:
                requests.get(f"{self.oai_host}/")
                break
            except Exception as e:
                if not wait_started:
                    Logger.log(f"{self.get_identifier()} is offline, waiting for it to become online...")
                    Logger.log(str(e), True)
                    wait_started = True
                time.sleep(1)
                continue

    def _abort(self):
        pass

    def _convert_data(self, data: dict, stream: bool = False) -> dict:
        def rename_dict_key(lhs: str, rhs: str):
            if lhs in data:
                data[rhs] = data[lhs]
                del data[lhs]
        rename_dict_key("max_tokens", "max_tokens")
        rename_dict_key("max_length", "max_tokens")
        rename_dict_key("rep_pen", "repeat_penalty")
        rename_dict_key("rep_pen_range", "repeat_last_n")
        rename_dict_key("sampler_seed", "seed")
        rename_dict_key("stop_sequence", "stop")
        return data

    @override
    def _generate_once(self, data: dict) -> Iterator[str]:
        data = self._convert_data(data)
        self._abort()

        for _ in range(5):
            try:
                response = requests.post(f"{self.oai_host}/v1/completions", data=json.dumps(data), stream=True, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'})
                client = sseclient.SSEClient(response)  # type: ignore
            except Exception as e:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} is offline.")
                Logger.log(str(e), True)
                exit(-1)

            if response.status_code == 503: # Server busy.
                Logger.log(f"{self.get_identifier()} is busy, trying again in ten seconds...", True)
                time.sleep(10)
                continue

            if response.status_code != 200:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} returned an error. HTTP status code: {response.status_code}")
                exit(-1)

            lastEvent = None
            try:
                for event in client.events():
                    if event.event == "message":
                        lastEvent = jsonText = event.data
                        if jsonText == "[DONE]":
                            break
                        data = json.loads(jsonText)
                        #logprobs = data['choices'][0].get('logprobs', {}).get('top_logprobs', [{}])[0].items() if 'logprobs' in data['choices'][0] else []
                        #probs = [{'tok_str': tok, 'prob': math.exp(logprob)} for tok, logprob in logprobs]
                        yield data['choices'][0]['text']
                break
            except Exception as e:
                Logger.log_event("Warning", Fore.YELLOW, f"{self.get_identifier()} returned an invalid response. Error while parsing {repr(lastEvent)}: {e}", True)
                continue
    
    @override
    def generate_batch(self, prompts: list[str], batch_size: int = 1, max_tokens: int = 8) -> list[str]:
        raise NotImplementedError()