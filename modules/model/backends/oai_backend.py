from typing import Iterator
from typing_extensions import override
from modules.model import LanguageModel
from modules.log import Logger
from colorama import Fore
import concurrent.futures
from functools import partial
import requests
import json
import time
import sseclient

__all__ = ("OaiModel",)


class OaiModel(LanguageModel):
    def __init__(self, oai_host: str, api_key: str, model: str, max_context: int, extra_api_params: dict = {}, extra_api_headers: dict = {}, auxiliary: bool = False):
        assert(isinstance(oai_host, str))
        if not oai_host:
            if not auxiliary:
                Logger.log_event("Error", Fore.RED, "Specify the model backend host using the argument --host.")
            else:
                Logger.log_event("Error", Fore.RED, "Specify the auxiliary model backend host using the argument --auxiliary-host.")
            exit(-1)
        self.oai_host = oai_host.strip('/')
        if not self.oai_host.startswith("http"):
            self.oai_host = f"http://{self.oai_host}"
        self.api_key = api_key
        self.model = model
        self.extra_api_params = extra_api_params
        self.extra_api_headers = extra_api_headers
        super().__init__(max_context, auxiliary)

    @override
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

    def _convert_data(self, data: dict) -> dict:
        def rename_dict_key(lhs: str, rhs: str):
            if lhs in data:
                data[rhs] = data[lhs]
                del data[lhs]
        def remove_dict_key(key: str):
            if key in data:
                del data[key]
        rename_dict_key("max_tokens", "max_tokens")
        rename_dict_key("max_length", "max_tokens")
        if "api.openai.com" in self.oai_host:
            remove_dict_key("rep_pen")
            remove_dict_key("rep_pen_range")
            remove_dict_key("max_context_length")
        else:
            rename_dict_key("rep_pen", "repetition_penalty")
            rename_dict_key("rep_pen_range", "repetition_penalty_range")
        rename_dict_key("sampler_seed", "seed")
        rename_dict_key("stop_sequence", "stop")
        if self.model and "claude" in self.model and "stop" in data:
            data["stop"] = [stop for stop in data["stop"] if stop.strip()]
        return data

    @override
    def _generate_token(self, data: dict) -> Iterator[str]:
        self._abort()

        data = self._convert_data(data)
        if self.model:
            data.update({
                "model": self.model
            })
        if self.extra_api_params:
            data.update(self.extra_api_params)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        if self.extra_api_headers:
            headers.update(self.extra_api_headers)

        for _ in range(5):
            try:
                response = requests.post(f"{self.oai_host}/v1/completions", data=json.dumps(data), stream=True, headers=headers)
                client = sseclient.SSEClient(response)  # type: ignore
            except Exception as e:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} is offline.")
                Logger.log(str(e), True)
                exit(-1)

            if response.status_code == 503: # Server busy.
                Logger.log(f"{self.get_identifier()} is busy, trying again in 3 seconds...", True)
                time.sleep(3)
                continue

            if response.status_code != 200:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} returned an error. HTTP status code: {response.status_code}\n{response.text}")
                exit(-1)

            lastEvent = None
            try:
                for event in client.events():
                    if event.event == "message":
                        lastEvent = jsonText = event.data
                        if jsonText == "[DONE]":
                            break
                        jsonData = json.loads(jsonText)

                        finishReason = jsonData['choices'][0]['finish_reason']
                        if finishReason is not None and finishReason != 'null':
                            if jsonData['choices'][0]['text']:
                                yield jsonData['choices'][0]['text']
                            break

                        #logprobs = jsonData['choices'][0].get('logprobs', {}).get('top_logprobs', [{}])[0].items() if 'logprobs' in jsonData['choices'][0] else []
                        #probs = [{'tok_str': tok, 'prob': math.exp(logprob)} for tok, logprob in logprobs]
                        yield jsonData['choices'][0]['text']
                        lastEvent = None
                break
            except Exception as e:
                Logger.log_event("Warning", Fore.YELLOW, f"{self.get_identifier()} returned an invalid response. Error while parsing {repr(lastEvent)}: {e}", True)
                continue

    @override
    def _generate_token_chat(self, data: dict) -> Iterator[str]:
        self._abort()

        data = self._convert_data(data)
        if self.model:
            data.update({
                "model": self.model
            })
        if self.extra_api_params:
            data.update(self.extra_api_params)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        if self.extra_api_headers:
            headers.update(self.extra_api_headers)

        for _ in range(5):
            try:
                response = requests.post(f"{self.oai_host}/v1/chat/completions", data=json.dumps(data), stream=True, headers=headers)
                client = sseclient.SSEClient(response)  # type: ignore
            except Exception as e:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} is offline.")
                Logger.log(str(e), True)
                exit(-1)

            if response.status_code == 503: # Server busy.
                Logger.log(f"{self.get_identifier()} is busy, trying again in 3 seconds...", True)
                time.sleep(3)
                continue

            if response.status_code != 200:
                Logger.log_event("Error", Fore.RED, f"{self.get_identifier()} returned an error. HTTP status code: {response.status_code}\n{response.text}")
                exit(-1)

            lastEvent = None
            try:
                for event in client.events():
                    if event.event == "message":
                        lastEvent = jsonText = event.data
                        if jsonText == "[DONE]":
                            break
                        jsonData = json.loads(jsonText)

                        finishReason = jsonData['choices'][0]['finish_reason']
                        if finishReason is not None and finishReason != 'null':
                            break

                        #logprobs = jsonData['choices'][0].get('logprobs', {}).get('top_logprobs', [{}])[0].items() if 'logprobs' in jsonData['choices'][0] else []
                        #probs = [{'tok_str': tok, 'prob': math.exp(logprob)} for tok, logprob in logprobs]
                        yield jsonData['choices'][0]['delta']['content']
                        lastEvent = None
                break
            except Exception as e:
                Logger.log_event("Warning", Fore.YELLOW, f"{self.get_identifier()} returned an invalid response. Error while parsing {repr(lastEvent)}: {e}", True)
                continue
    
    def supports_batching(self) -> bool:
        return True
    
    @override
    def generate_batch(self, prompts: list[str], max_tokens: int = 8, stop_sequences: list[str] = []) -> list[str]:
        def gen_prompt(prompt: str, idx: int) -> tuple[str, int]:
            return self.generate(prompt, max_tokens, stop_sequences), idx
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            process_func = partial(gen_prompt)
            future_to_idx = {executor.submit(process_func, prompt, idx): idx 
                            for idx, prompt in enumerate(prompts)}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                result, idx = future.result()
                results.append((idx, result))

        results.sort(key=lambda x: x[0])
        return [prompt for _, prompt in results]