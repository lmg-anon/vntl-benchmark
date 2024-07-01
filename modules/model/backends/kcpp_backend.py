from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore
import requests
import json
import time
import sseclient

__all__ = ("KcppModel",)


class KcppModel(LanguageModel):
    def __init__(self, kcpp_host: str, max_context: int, auxiliary: bool = False):
        assert(isinstance(kcpp_host, str))
        if not kcpp_host:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model backend host using the argument --host.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model backend host using the argument --auxiliary-host.")
            exit(-1)
        self.kcpp_host = kcpp_host.strip('/')
        if not self.kcpp_host.startswith("http"):
            self.kcpp_host = f"http://{self.kcpp_host}"
        super().__init__(max_context, auxiliary)

    @override
    def wait(self):
        wait_started = False
        while True:
            try:
                requests.get(f"{self.kcpp_host}/")
                break
            except Exception as e:
                if not wait_started:
                    Logger.log(f"{self.get_identifier()} is offline, waiting for it to become online...")
                    Logger.log(str(e), True)
                    wait_started = True
                time.sleep(1)
                continue

    @override
    def _abort(self):
        try:
            requests.post(f"{self.kcpp_host}/api/extra/abort")
        except Exception as e:
            Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} is offline.")
            Logger.log(str(e), True)
            exit(-1)

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        self._abort()

        for _ in range(5):
            try:
                response = requests.post(f"{self.kcpp_host}/api/extra/generate/stream", data=json.dumps(data), stream=True, headers={'Content-Type': 'application/json'})
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

            try:
                for event in client.events():
                    if event.event == "message":
                        data = json.loads(event.data)
                        yield data["token"]
                break
            except Exception as e:
                Logger.log_event("Warning", Fore.YELLOW, f"{self.get_identifier()} returned an invalid response. Error while parsing: {e}", True)
                continue
    
    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8, stop_sequences: list[str] = []) -> list[str]:
        raise NotImplementedError()