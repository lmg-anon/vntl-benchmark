import subprocess
import yaml
import signal
import sys
import shutil
import os
from dataclasses import dataclass

current_process = None
current_auxiliary_process = None

@dataclass
class ModelParams:
    backend: str
    backend_host: str
    backend_path: str
    backend_api_key: str
    backend_extra_args: str
    path: str
    prompt_format: str
    preset: str

def find_program_path(program_name):
    """
    Find the path of a program in the system's PATH environment variable.
    Returns None if the program is not found.
    """
    return shutil.which(program_name)

def get_run_command(model: ModelParams, context_size: int, thread_number: int, port: int) -> str:
    if model.backend == "ooba":
        command = f"\"{sys.executable}\" {model.backend_path} --model \"{model.path}\" --n_ctx {context_size} --max_seq_len {context_size} --compress_pos_emb {context_size // 2048} --threads {thread_number} --api --api-blocking-port {port} {model.backend_extra_args}"
    elif model.backend == "koboldcpp":
        command = f"{model.backend_path} --model \"{model.path}\" --contextsize {context_size} --threads {thread_number} --stream --port {port} {model.backend_extra_args}"
    elif model.backend == "llamacpp":
        command = f"{model.backend_path} -m \"{model.path}\" -t {thread_number} -c {context_size} --port {port} {model.backend_extra_args}"
    else:
        return ""
    return command

def run_python_script(title: str, model: ModelParams, auxiliary_model: ModelParams | None, context_size: int, extra_args: str):
    command = f"\"{sys.executable}\" main.py --title \"{title}\" --backend {model.backend} --context {context_size} --preset {model.preset}"

    if model.prompt_format:
        command += f" --format {model.prompt_format}"
    if model.backend in ["koboldcpp", "llamacpp", "ooba", "openai"]:
        command += f" --host {model.backend_host}"
        if model.backend_api_key:
            command += f" --api-key {model.backend_api_key}"
    elif model.backend == "llamapy":
        command += f" --model {model.path}"

    if auxiliary_model:
        command += f" --auxiliary-backend {auxiliary_model.backend} --auxiliary-preset {auxiliary_model.preset}"

        if auxiliary_model.prompt_format:
            command += f" --auxiliary-format {auxiliary_model.prompt_format}"
        if auxiliary_model.backend in ["koboldcpp", "llamacpp", "ooba", "openai"]:
            command += f" --auxiliary-host {auxiliary_model.backend_host}"
            if auxiliary_model.backend_api_key:
                command += f" --auxiliary-api-key {auxiliary_model.backend_api_key}"
        elif auxiliary_model.backend == "llamapy":
            command += f" --auxiliary-model {auxiliary_model.path}"

    if extra_args:
        command += f" {extra_args}"

    subprocess.run(command, shell=True)

def exit_gracefully(signum, frame):
    global current_process
    if current_process is not None and current_process.poll() is None:
        current_process.terminate()
        current_process.wait()
        current_process = None
    global current_auxiliary_process
    if current_auxiliary_process is not None and current_auxiliary_process.poll() is None:
        current_auxiliary_process.terminate()
        current_auxiliary_process.wait()
        current_auxiliary_process = None
    exit()

def run_plan(run_plan_file):
    global current_process
    global current_auxiliary_process

    with open(run_plan_file, "r") as f:
        run_plan = yaml.safe_load(f)

    signal.signal(signal.SIGINT, exit_gracefully)

    for item in run_plan:
        model_item = item["model"]

        model_backend = model_item["backend"]
        if model_backend not in ["koboldcpp", "llamacpp", "llamapy", "ooba", "openai"]:
            print(f"Invalid model_backend: {model_backend}")
            continue

        model = ModelParams(
            model_backend,
            model_item.get("backend_host", ""),
            model_item.get("backend_path", ""),
            model_item.get("backend_api_key", ""),
            model_item.get("backend_extra_args", ""),
            model_item.get("path", ""),
            model_item.get("prompt_format", ""),
            model_item.get("preset", "default")
        )

        auxiliary_model = None

        aux_model_item = item.get("auxiliary_model")
        if aux_model_item:
            auxiliary_model_backend = aux_model_item.get("backend", "")
            if auxiliary_model_backend not in ["koboldcpp", "llamacpp", "llamapy", "ooba", "openai"]:
                print(f"Invalid auxiliary_model_backend: {auxiliary_model_backend}")
                continue

            auxiliary_model = ModelParams(
                auxiliary_model_backend,
                aux_model_item.get("backend_host", ""),
                aux_model_item.get("backend_path", ""),
                aux_model_item.get("backend_api_key", ""),
                aux_model_item.get("backend_extra_args", ""),
                aux_model_item.get("path", ""),
                aux_model_item.get("prompt_format", ""),
                aux_model_item.get("preset", "default")
            )

        title = item.get("title", None)
        context_size = item.get("context_size", 2048)
        thread_number = item["thread_number"]
        extra_args = item.get("extra_args", "")

        if not model.backend_host:
            model.backend_host = "127.0.0.1:5000"
            run_command = get_run_command(model, context_size, thread_number, 5000)
            if run_command:
                if current_process is not None and current_process.poll() is None:
                    # Terminate the current process if the parameters don't match
                    if current_process.args != run_command:
                        current_process.terminate()
                        current_process.wait()
                        current_process = subprocess.Popen(run_command, cwd=os.path.dirname(os.path.realpath(model.backend_path)), shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    current_process = subprocess.Popen(run_command, cwd=os.path.dirname(os.path.realpath(model.backend_path)), shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if auxiliary_model:
            if not auxiliary_model.backend_host:
                auxiliary_model.backend_host = "127.0.0.1:5001"
                run_command = get_run_command(auxiliary_model, 2048, thread_number, 5001)
                if run_command:
                    if current_auxiliary_process is not None and current_auxiliary_process.poll() is None:
                        # Terminate the current process if the parameters don't match
                        if current_auxiliary_process.args != run_command:
                            current_auxiliary_process.terminate()
                            current_auxiliary_process.wait()
                            current_auxiliary_process = subprocess.Popen(run_command, cwd=os.path.dirname(os.path.realpath(auxiliary_model.backend_path)), shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        current_auxiliary_process = subprocess.Popen(run_command, cwd=os.path.dirname(os.path.realpath(auxiliary_model.backend_path)), shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        run_python_script(title, model, auxiliary_model, context_size, extra_args)

    # If the script is exiting, terminate the current process
    if current_process is not None and current_process.poll() is None:
        current_process.terminate()
        current_process.wait()
        current_process = None

    if current_auxiliary_process is not None and current_auxiliary_process.poll() is None:
        current_auxiliary_process.terminate()
        current_auxiliary_process.wait()
        current_auxiliary_process = None

if __name__ == "__main__":
    run_plan_file = "run_plan.yaml"
    run_plan(run_plan_file)