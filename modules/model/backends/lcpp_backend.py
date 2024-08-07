from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore
import requests
import json
import time

__all__ = ("LcppModel",)


class LcppModel(LanguageModel):
    def __init__(self, lcpp_host: str, max_context: int, auxiliary: bool = False):
        assert(isinstance(lcpp_host, str))
        if not lcpp_host:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model backend host using the argument --host.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model backend host using the argument --auxiliary-host.")
            exit(-1)
        self.lcpp_host = lcpp_host.strip('/')
        if not self.lcpp_host.startswith("http"):
            self.lcpp_host = f"http://{self.lcpp_host}"
        super().__init__(max_context, auxiliary)

    @override
    def wait(self):
        wait_started = False
        while True:
            try:
                response = requests.get(f"{self.lcpp_host}/")
                break
            except Exception as e:
                if not wait_started:
                    Logger.log(f"{self.get_identifier()} is offline, waiting for it to become online...")
                    Logger.log(str(e), True)
                    wait_started = True
                time.sleep(1)
                continue

    def _convert_data(self, data: dict, stream: bool = False) -> dict:
        def rename_dict_key(lhs: str, rhs: str):
            if lhs in data:
                data[rhs] = data[lhs]
                del data[lhs]
        rename_dict_key("max_context_length", "n_ctx")
        rename_dict_key("max_tokens", "n_predict")
        rename_dict_key("max_length", "n_predict")
        rename_dict_key("rep_pen", "repeat_penalty")
        rename_dict_key("rep_pen_range", "repeat_last_n")
        rename_dict_key("tfs", "tfs_z")
        rename_dict_key("typical", "typical_p")
        rename_dict_key("sampler_seed", "seed")
        rename_dict_key("stop_sequence", "stop")
        if "stop" not in data:
            data["stop"] = []
        # TODO: Stream support.
        if "stream" in data:
            del data["stream"]
        data["n_keep"] = -1
        return data

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        data = self._convert_data(data)
        response_text = str()

        for _ in range(5):
            if response_text:
                break

            try:
                response = requests.post(f"{self.lcpp_host}/completion", data=json.dumps(data), headers={'Content-Type': 'application/json'})
            except Exception as e:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} is offline.")
                Logger.log(str(e), True)
                exit(-1)

            if response.status_code == 503 or response.status_code == 400: # Server busy.
                Logger.log(f"{self.get_identifier()} is busy, trying again in ten seconds...", True)
                time.sleep(10)
                continue

            if response.status_code != 200:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} returned an error. HTTP status code: {response.status_code}")
                exit(-1)

            try:
                response_dict = response.json()
                response_text = response_dict["content"]
                if not response_text and response_dict["stopped_eos"]:
                    return ""
            except Exception as e:
                Logger.log_event("Warning", Fore.YELLOW, f"{self.get_identifier()} returned an invalid response. Error while parsing: {e}", True)
                continue

        return response_text
    
    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        raise NotImplementedError()
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8, stop_sequences: list[str] = []) -> list[str]:
        raise NotImplementedError()