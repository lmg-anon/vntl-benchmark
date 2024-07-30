import subprocess
import yaml
import json
import signal
import sys
import shutil
import os
import psutil
import threading
import time
import argparse
import re
from queue import Queue
from dataclasses import dataclass

current_process = None

download_queue = Queue()
downloading_models = set()
downloaded_models = set()

@dataclass
class Config:
    concurrent_downloads: int = 0
    model_cache_size: int = 0
    extra_args: str = ""
    verbose: bool = False
    clean: bool = False
    allow_resume: bool = False
    model_name_regex: str = ""
    base_downloader: dict = None
    base_model: dict = None
    custom_backends: dict = None

@dataclass
class DownloaderParams:
    repo: str
    output: str
    branch: str = "main"
    filters: str = ""
    token: str = ""
    delete_after: bool = False

@dataclass
class BackendParams:
    type: str
    host: str = ""
    api_key: str = ""
    model: str = ""
    chat_endpoint: bool = False
    batch_endpoint: bool = False
    extra_api_params: dict = None
    extra_api_headers: dict = None

@dataclass
class RunParams:
    path: str
    arguments: str

@dataclass
class ModelParams:
    seed: int | None = None
    context_size: int = 1024
    batch_size: int = 1
    thread_number: int = 5
    prompt_format: str = ""
    preset: str = ""
    stop: dict = None
    prefill: str = ""
    eos_token: str = ""

@dataclass
class Dataset:
    name: str
    results_path: str
    dataset_path: str
    samples: int
    samples_fd: list[str]

def find_program_path(program_name):
    """
    Find the path of a program in the system's PATH environment variable.
    Returns None if the program is not found.
    """
    return shutil.which(program_name)

def get_run_command(run: RunParams | None, model: ModelParams, backend: BackendParams, port: int) -> str:
    if backend.type == "ooba":
        command = f"\"{sys.executable}\" \"{run.path}\" --model \"{backend.model}\" --n_ctx {model.context_size} --max_seq_len {model.context_size} --compress_pos_emb {model.context_size // 2048} --threads {model.thread_number} --api --api-blocking-port {port} {run.arguments}"
    elif backend.type == "koboldcpp":
        command = f"\"{run.path}\" --model \"{backend.model}\" --contextsize {model.context_size} --threads {model.thread_number} --stream --port {port} {run.arguments}"
    elif backend.type == "llamacpp":
        command = f"\"{run.path}\" -m \"{backend.model}\" -t {model.thread_number} -c {model.context_size} --port {port} {run.arguments}"
    else:
        command = f"\"{run.path}\" {run.arguments.format(model_path = backend.model, threads = model.thread_number, context_size = model.context_size, port = port)}"
    return command

def run_python_script(title: str, model: ModelParams, backend: BackendParams, dataset: Dataset, extra_args: str):
    command = f"\"{sys.executable}\" main.py --y --title \"{title}\" --backend {backend.type} --context-size {model.context_size} --batch-size {model.batch_size}"

    if model.seed:
        command += f" --seed {model.seed}"
    if model.prompt_format:
        command += f" --format \"{model.prompt_format}\""
    if model.preset:
        command += f" --preset \"{model.preset}\""
    if model.stop:
        stop_sequences = json.dumps(model.stop).replace('"', '\\"')
        command += f" --stop-sequences \"{stop_sequences}\""
    if model.prefill:
        command += f" --prefill \"{model.prefill}\""
    if model.eos_token:
        command += f" --eos-token \"{model.eos_token}\""

    if backend.type in ["koboldcpp", "llamacpp", "ooba", "openai"]:
        command += f" --host {backend.host}"
        if backend.api_key:
            command += f" --api-key {backend.api_key}"
        if backend.model:
            command += f" --model \"{backend.model}\""
        if backend.batch_endpoint:
            command += f" --batch-api"
        if backend.chat_endpoint:
            command += f" --chat-api"
        if backend.extra_api_params:
            extra_api_params = json.dumps(backend.extra_api_params).replace('"', '\\"')
            command += f" --extra-api-params \"{extra_api_params}\""
        if backend.extra_api_headers:
            extra_api_headers = json.dumps(backend.extra_api_headers).replace('"', '\\"')
            command += f" --extra-api-headers \"{extra_api_headers}\""
    elif backend.type in ["llamapy", "sugoi", "tlservice"]:
        command += f" --model {backend.model}"

    command += f" --results-path \"{dataset.results_path}\""
    command += f" --dataset-path \"{dataset.dataset_path}\""
    command += f" --samples {dataset.samples}"
    command += " --samples-fd \"%s\"" % json.dumps(dataset.samples_fd).replace('"', '\\"')
    
    if extra_args:
        command += f" {extra_args}"

    subprocess.run(command, shell=True)

def kill_current_process():
    global current_process
    if current_process is not None and current_process.poll() is None:
        parent = psutil.Process(current_process.pid)
        for child in parent.children(recursive=True):
            child.kill()
            child.wait()
        current_process.terminate()
        current_process.wait()
        current_process = None

def exit_gracefully(signum, frame):
    kill_current_process()
    exit()

def run_plan(dataset: Dataset, config: Config):
    global current_process

    files = os.listdir("configs")
    files = [(os.path.join("configs", file_name), file_name) for file_name in files if file_name.endswith('.yml')]

    run_plan = []
    for run_plan_file, file_name in files:
        if file_name.startswith("_"):
            continue
        with open(run_plan_file, "r") as f:
            yamlDict = yaml.safe_load(f)
            yamlDict["title"] = file_name.rsplit(".", 1)[0]
            run_plan.append(yamlDict)

    signal.signal(signal.SIGINT, exit_gracefully)

    for i, item in enumerate(run_plan):
        title = item.get("title", None)

        if config.model_name_regex:
            if not re.search(config.model_name_regex, title, re.IGNORECASE):
                continue

        if config.base_downloader and "downloader" in item:
            for key, value in config.base_downloader.items():
                if key not in item["downloader"] or not item["downloader"][key]:
                    item["downloader"][key] = value

        if item["backend"]["type"] in config.custom_backends:
            base_backend = config.custom_backends[item["backend"]["type"]]
            for key, value in base_backend.items():
                if key not in item["backend"] or not item["backend"][key]:
                    item["backend"][key] = value
            item["backend"]["type"] = base_backend["type"]

        if config.base_model:
            if "model" not in item:
                item["model"] = {}
            for key, value in config.base_model.items():
                if key not in item["model"]: #or not item["model"][key]:
                    item["model"][key] = value

        downloader = DownloaderParams(**item["downloader"]) if "downloader" in item else None
        backend = BackendParams(**item["backend"])
        run = RunParams(**item["run"]) if "run" in item else None
        model = ModelParams(**item["model"]) if "model" in item else ModelParams()

        if backend.type not in ["koboldcpp", "llamacpp", "llamapy", "ooba", "openai", "tlservice", "sugoi"]:
            print(f"Invalid backend: {backend.type}")
            continue
            
        print(f"• {title}")

        clean = item.get("clean", config.clean)
        allow_resume = item.get("allow_resume", config.allow_resume)
        if os.path.isfile(os.path.join(dataset.results_path, f"{title}.jsonl")):
            if clean:
                try:
                    os.remove(os.path.join(dataset.results_path, f"{title}.jsonl"))
                except:
                    pass
            elif not allow_resume:
                print("Result file exists, if you want to resume this run add `allow_resume: true` to the run plan.")
                continue

        extra_args = item.get("extra_args", "")
        extra_args = config.extra_args if not extra_args else f"{config.extra_args} {extra_args}"

        if downloader is not None:
            if os.path.isdir(downloader.output):
                downloader.delete_after = False
            else:
                os.mkdir(downloader.output)

            if downloader.output not in downloaded_models:
                print("Waiting for download...")
                if downloader.output not in downloading_models:
                    download_queue.put(downloader)

                while downloader.output not in downloaded_models:
                    # Wait model to get downloaded.
                    time.sleep(1)
            else:
                print("Model already downloaded, skipping download.")

        if run is not None:
            if backend.type in ["ooba", "koboldcpp", "llamacpp"]:
                backend.host = "127.0.0.1:5000"
            run_command = get_run_command(run, model, backend, 5000)
            if current_process is not None and current_process.poll() is None:
                # Terminate the current process if the parameters don't match
                if current_process.args != run_command:
                    kill_current_process()
                    current_process = subprocess.Popen(run_command, cwd=os.path.dirname(os.path.realpath(run.path)), shell=config.verbose, stdout=None if config.verbose else subprocess.DEVNULL, stderr=None if config.verbose else subprocess.DEVNULL)
            else:
                current_process = subprocess.Popen(run_command, cwd=os.path.dirname(os.path.realpath(run.path)), shell=config.verbose, stdout=None if config.verbose else subprocess.DEVNULL, stderr=None if config.verbose else subprocess.DEVNULL)

            # Check if the process has started correctly or exited with an error code
            try:
                stdout, stderr = current_process.communicate(timeout=1)
                if current_process.returncode != 0:
                    print(f"Process failed to start or exited with an error code: {current_process.returncode}")
                    if stderr is not None:
                        print(f"Error: {stderr.decode().strip()}")
                    continue
            except:
                pass

        run_python_script(title, model, backend, dataset, extra_args)

        if downloader is not None:
            downloaded_models.remove(downloader.output)
            if downloader.delete_after:
                shutil.rmtree(downloader.output)

    # If the script is exiting, terminate the current process
    kill_current_process()

def download_model(config: Config, downloader: DownloaderParams):
    repo = downloader.repo
    if downloader.filters:
        repo += f":{downloader.filters}"

    command = f"\"{sys.executable}\" hf_downloader.py \"{repo}\" --output \"{downloader.output}\" --branch \"{downloader.branch}\""
    if not config.verbose:
        command += " --silent"
    if downloader.token:
        command += f" --token \"{downloader.token}\""
    subprocess.run(command, shell=True)
    downloaded_models.add(downloader.output)
    downloading_models.remove(downloader.output)

def download_models_concurrently(config: Config):
    while True:
        if len(downloaded_models) < config.model_cache_size + 1:
            if len(downloading_models) < config.concurrent_downloads:
                downloader = download_queue.get()
                if downloader.output not in downloaded_models:
                    downloading_models.add(downloader.output)
                    threading.Thread(target=download_model, args=(config, downloader,)).start()
            else:
                time.sleep(1)  # Wait for a short interval before checking again
        else:
            time.sleep(1)  # Wait for a short interval before checking again

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VNTL Benchmark")

    parser.add_argument("--model", type=str, help="model name regex", default="")

    parser.add_argument("--clean", action="store_true", help="enable clean run")
    parser.add_argument("--allow-resume", action="store_true", help="enable resume run")

    parser.add_argument("--dataset", type=str, help="dataset name", required=True)
    parser.add_argument("--results-path", type=str, help="results directory path", default="")
    parser.add_argument("--dataset-path", type=str, help="dataset file path", default="")
    parser.add_argument("--samples", type=int, help="number of samples for evaluation (default: 128)", default=128)
    parser.add_argument("--samples-fd", type=str, help="filter samples by fidelity", default="")

    parser.add_argument("--extra-args", type=str, help="extra args to run main script", default="")
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")

    #parser.add_argument("--y", action="store_true", help="")

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    with open("datasets.yml", "r") as f:
        datasets = [Dataset(**dataset) for dataset in yaml.safe_load(f)]

    dataset = next((dataset for dataset in datasets if dataset.name == args.dataset), None)

    if dataset is None:
        print("Unknown dataset.")
        exit(-1)

    dataset.results_path = args.results_path if args.results_path else dataset.results_path
    dataset.dataset_path = args.dataset_path if args.dataset_path else dataset.dataset_path
    dataset.samples = args.samples if args.samples else dataset.samples
    dataset.samples_fd = args.samples_fd if args.samples_fd else dataset.samples_fd
    
    with open("config.yml", "r") as f:
        config = Config(**yaml.safe_load(f))

    config.extra_args = args.extra_args if not config.extra_args else f"{config.extra_args} {args.extra_args}"
    config.verbose = args.verbose if args.verbose else config.verbose
    config.clean = args.clean if args.clean else config.clean
    config.allow_resume = args.allow_resume if args.allow_resume else config.allow_resume
    config.model_name_regex = args.model if not config.model_name_regex else f"{config.model_name_regex}|{args.model}"

    download_thread = threading.Thread(target=download_models_concurrently, args=(config,))
    download_thread.daemon = True
    download_thread.start()

    run_plan(dataset, config)