import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings

from colorama import Fore, init as colorama_init
from modules.model import LanguageModel
from modules.model.backends import KcppModel, LcppModel, OobaModel, OaiModel, LPY_PRESENT, EXL2_PRESENT, TF_PRESENT, UNSLOTH_PRESENT, TRANSLATORS_PRESENT, SUGOI_PRESENT
if LPY_PRESENT:
    from modules.model.backends import LpyModel
if EXL2_PRESENT:
    from modules.model.backends import EXL2Model
if TF_PRESENT:
    from modules.model.backends import TFModel
if UNSLOTH_PRESENT:
    from modules.model.backends import UnslothModel
if TRANSLATORS_PRESENT:
    from modules.model.backends import TLServiceModel
if SUGOI_PRESENT:
    from modules.model.backends import SugoiModel
from modules.prompt.formats import *
from modules.visual_novel import *
from modules.translation import *
from modules.log import Logger
from tqdm import tqdm
import argparse
import json
import time
import os
import re
import random
import statistics
import sacrebleu


st_model = None
def get_similarity_batched(texts1, texts2):
    import torch
    from sentence_transformers import SentenceTransformer, util
    global st_model
    if st_model is None:
        #paraphrase-multilingual-mpnet-base-v2
        #all-MiniLM-L12-v2
        #all-distilroberta-v1
        #all-mpnet-base-v2
        #all-MiniLM-L6-v2
        st_model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu', cache_folder="./s_cache")
    embeddings1 = st_model.encode(texts1, convert_to_tensor=True, show_progress_bar=False)
    embeddings2 = st_model.encode(texts2, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.diag()

def get_similarity(text1, text2):
    text1 = text1.strip("っ。～…―（）「」｢｣『』“”„\"'`，、○.,()«»~ \t\r\n")
    text2 = text2.strip("っ。～…―（）「」｢｣『』“”„\"'`，、○.,()«»~ \t\r\n")
    if text1.lower() == text2.lower():
        return 1.0
    return float(get_similarity_batched([text1], [text2])[0])

def get_bleu(ref, hyp):
    ref = ref.strip("っ。～…―（）「」｢｣『』“”„\"'`，、○.,()«»~ \t\r\n")
    hyp = hyp.strip("っ。～…―（）「」｢｣『』“”„\"'`，、○.,()«»~ \t\r\n")
    if ref.lower() == hyp.lower():
        return 100
    bleu = sacrebleu.sentence_bleu(hyp, [ref])
    return bleu.score

def get_chrf(ref, hyp):
    ref = ref.strip("っ。～…―（）「」｢｣『』“”„\"'`，、○.,()«»~ \t\r\n")
    hyp = hyp.strip("っ。～…―（）「」｢｣『』“”„\"'`，、○.,()«»~ \t\r\n")
    if ref.lower() == hyp.lower():
        return 100
    chrf = sacrebleu.sentence_chrf(hyp, [ref])
    return chrf.score

def detect_high_repetition(line, threshold=25):
    def is_repetition(substring, fullstring):
        count = fullstring.count(substring)
        return count >= threshold
    for i in range(len(line)):
        for j in range(i+3, len(line)):
            substring = line[i:j]
            if len(substring) > 1 and is_repetition(substring, line):
                return line
    return None

if __name__ == "__main__":
    SCORE_VERSION = 2
    MAX_ENTRIES_PER_BATCH = 10
    MAX_NEW_TOKENS = 120
    MAX_BENCHMARK_OUTPUT = 128

    # # 'TLAssist-test-val.txt'
    # # 'TLAssist-validation-v4_SIMILARITY_NS.txt'
    # # 'TLAssist-all-v4_SIMILARITY_NS.txt'
    # # 'TLAssist-validation-v5_SIMILARITY.jsonl'
    # # 'Mashiro_full.jsonl'
    # EVAL_DATASET = 'Mashiro_full.jsonl' #'TLAssist-validation-v4_SIMILARITY_NS.txt'

    # RESULTS_DIR = "results_mashiro" if "Mashiro" in EVAL_DATASET else "results"

    SUPPORTED_BACKENDS = {"koboldcpp", "llamacpp", "ooba", "llamapy", "openai", "exl2", "transformers", "unsloth", "tlservice", "sugoi"}
    if not LPY_PRESENT:
        SUPPORTED_BACKENDS.remove("llamapy")
    if not EXL2_PRESENT:
        SUPPORTED_BACKENDS.remove("exl2")
    if not TF_PRESENT:
        SUPPORTED_BACKENDS.remove("transformers")
    if not UNSLOTH_PRESENT:
        SUPPORTED_BACKENDS.remove("unsloth")
    if not TRANSLATORS_PRESENT:
        SUPPORTED_BACKENDS.remove("tlservice")
    if not SUGOI_PRESENT:
        SUPPORTED_BACKENDS.remove("sugoi")

    parser = argparse.ArgumentParser(description="VNTL Benchmark")

    parser.add_argument("--title", type=str, help="run title")

    parser.add_argument("--backend", type=str, choices=SUPPORTED_BACKENDS, help="model backend type")
    parser.add_argument("--context-size", type=int, help="model context size (default: 2048)", default=2048)
    parser.add_argument("--batch-size", type=int, help="model batch size (default: 1)", default=1)
    parser.add_argument("--preset", type=str, help="model preset (default: default)")
    #parser.add_argument("--format", type=str, help="model prompt format (default: alpaca)")
    parser.add_argument("--prefill", type=str, help="model response prefill, used only in chat endpoints")
    parser.add_argument("--eos-token", type=str, help="model eos token, used only in completion endpoints (default: </s>)", default="</s>")
    parser.add_argument("--stop-sequences", type=str, help="model stop sequences", default="[]")
    parser.add_argument("--model", type=str, help="model path for Non-API backends, or model ID for API backends")
    parser.add_argument("--host", type=str, help="host of the model backend")
    parser.add_argument("--api-key", type=str, help="api key of the model backend")
    parser.add_argument("--chat-api", action="store_true", help="use chat completions endpoint, if available for the backend type")
    parser.add_argument("--batch-api", action="store_true", help="use batch endpoint, only available for openai")
    parser.add_argument("--batch-input-file", type=str, help="batch input file path to generate for the batch api")
    parser.add_argument("--batch-output-file", type=str, help="batch output file path to read for the batch api")
    parser.add_argument("--extra-api-params", type=str, help="extra parameters to send in the api request", default="{}")
    parser.add_argument("--extra-api-headers", type=str, help="extra headers to send in the api request", default="{}")

    parser.add_argument("--results-path", type=str, help="results directory path (default: ./results)", default="./results")
    parser.add_argument("--dataset-path", type=str, help="eval dataset path (default: ./TLAssist-validation-v4_SIMILARITY_NS.txt)", default="./TLAssist-validation-v4_SIMILARITY_NS.txt")

    parser.add_argument("--shuffle", action="store_true", help="enable prompt shuffling")

    parser.add_argument("--seed", type=int, help="initial rng seed (default: 3407)", default=3407)

    parser.add_argument("--verbose", action="store_true", help="enable verbose output")

    parser.add_argument("--y", action="store_true", help="")

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    colorama_init()
    Logger.init()

    if args.seed and args.seed >= 0:
        random.seed(args.seed)
        LanguageModel.base_seed = args.seed

    if args.verbose:
        Logger.print_verbose = True

    scores = []
    processed_prompts = []
    if os.path.isfile(os.path.join(args.results_path, f"{args.title}.jsonl")):
        while True:
            if args.y:
                reply = "y"
            else:
                reply = input("Do you want to resume the previous run? (Y/N): ").strip().lower()
            if reply == "y":
                with open(os.path.join(args.results_path, f"{args.title}.jsonl"), "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        obj = json.loads(line)
                        scores.append(obj["accuracy"])
                        processed_prompts.append(obj["prompt"])
                break
            elif reply == "n":
                try:
                    os.remove(os.path.join(args.results_path, f"{args.title}.jsonl"))
                except:
                    pass
                break

    if len(processed_prompts) >= MAX_BENCHMARK_OUTPUT:
        print("Already finished.")
        exit(0)

    extra_api_params = json.loads(args.extra_api_params)
    assert isinstance(extra_api_params, dict)

    extra_api_headers = json.loads(args.extra_api_headers)
    assert isinstance(extra_api_headers, dict)

    stop_sequences = json.loads(args.stop_sequences)
    assert isinstance(stop_sequences, list)

    if args.backend == "openai":
        model = OaiModel(args.host, args.api_key, args.model, args.context_size, extra_api_params, extra_api_headers)
        model.wait()
    elif args.backend == "llamacpp":
        assert not extra_api_params
        assert not extra_api_headers
        model = LcppModel(args.host, args.context_size)
        model.wait()
    elif args.backend == "tlservice":
        assert not extra_api_params
        assert not extra_api_headers
        model = TLServiceModel(args.model)
    elif args.backend == "sugoi":
        assert not extra_api_params
        assert not extra_api_headers
        model = SugoiModel(args.model)

    model.presets = {
        "temperature": 0,
        "rep_pen": 1,
    }

    if args.batch_input_file:
        try:
            os.remove(args.batch_input_file)
        except:
            pass

    batch_api_output = []
    if args.batch_output_file:
        with open(args.batch_output_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                obj = json.loads(line)
                batch_api_output.append(obj["response"]["body"]["choices"][0]["message"]["content"])

    # Parse the characters file
    books = VisualNovels("characters.txt")
    books.read_file()

    start_bench = time.time()

    tf = TranslationFile(args.dataset_path)
    tf.read_file()

    entries = tf.entries
    last_entry = None

    skipping = len(processed_prompts) > 0

    entry_batches: list[list[list[TranslationEntry]]] = [[]]
    metadata_batches: list[list[str]] = [[]]
    prompt_batches: list[list[str]] = [[]]
    for idx in range(len(entries)):
        batch = entries[max(idx-MAX_ENTRIES_PER_BATCH-1, 0):idx+1]
        if last_entry == batch[-1]:
            break
        last_entry = batch[-1]
        batch = [entry for entry in batch if entry.book_id == last_entry.book_id]

        if "Mashiro" in args.dataset_path:
            # For Mashiro dataset
            if "absolute" in batch[-1].fidelity or "high" in batch[-1].fidelity:
                continue
        else:
            # For SenrenBanka dataset
            if "absolute" not in batch[-1].fidelity:
                continue

        # We want the LLM to have some completion examples.
        if idx < 5:
            continue

        ID = batch[-1].book_id
        
        if isinstance(model, TLServiceModel) or isinstance(model, SugoiModel):
            metadata = None
            prompt = batch[-1].japanese.split("]: ", 1)[-1].split("】：", 1)[-1].split("】:", 1)[-1]
        else:
            block = ""
            for entry in batch:
                block += \
                    f"<<JAPANESE>>\n" \
                    f"{entry.japanese}\n" \
                    f"<<ENGLISH>>\n" \
                    f"{entry.english if entry != batch[-1] else ''}{args.eos_token if entry != batch[-1] else ''}\n"
            block = block.strip() + "\n"
            block = re.sub(r"【(.*?)】：", r"[\1]: ", block)
            
            def compile_metadata(characters):
                metadata_chars = []
                for character in characters:
                    metadata_section = f"[character] Name: {character.name} ({character.japanese_name}) | Gender: {character.gender}"
                    if character.aliases and character.aliases != "None":
                        metadata_section += f" | Aliases: {character.aliases}"
                    metadata_chars.append(metadata_section)

                metadata = ""
                if metadata_chars:
                    metadata += '<<METADATA>>\n' + '\n'.join(random.sample(metadata_chars, k=len(metadata_chars))) + "\n"
                return metadata

            # Find all unique speaking characters in this block
            speaking_characters: set[Character] = set()
            speaking_names = set(re.findall(r'\[(.*?)\]', block))
            for name in speaking_names:
                _character = books.get_character(ID, name)
                if _character:
                    speaking_characters.add(_character)
            metadata = compile_metadata(speaking_characters)
            prompt = metadata + "<<START>>\n" + block
        
        if skipping:
            if isinstance(model, TLServiceModel) or isinstance(model, SugoiModel):
                skipping = processed_prompts[-1] != prompt
            else:
                skipping = processed_prompts[-1][processed_prompts[-1].index("<<JAPANESE>>"):] != prompt[prompt.index("<<JAPANESE>>"):]
            continue

        entry_batches[-1].append(batch)
        metadata_batches[-1].append(metadata)
        prompt_batches[-1].append(prompt)
        if len(entry_batches[-1]) == args.batch_size:
            entry_batches.append([])
            metadata_batches.append([])
            prompt_batches.append([])

    if args.shuffle:
        random.Random(args.seed).shuffle(entry_batches)
        random.Random(args.seed).shuffle(metadata_batches)
        random.Random(args.seed).shuffle(prompt_batches)

    count = len(processed_prompts)
    pbar = tqdm(total=MAX_BENCHMARK_OUTPUT-len(processed_prompts))
    for batches, metadatas, prompts in zip(entry_batches, metadata_batches, prompt_batches):
        if count >= MAX_BENCHMARK_OUTPUT:
            break

        if not args.batch_input_file and not args.batch_output_file:
            tqdm.write("")
            tqdm.write("Generating...")

        inf_start = time.time()

        results = []
        if args.batch_input_file or args.batch_output_file or args.batch_api:
            if args.batch_output_file:
                assert batch_api_output
                results.append(batch_api_output[count])
            else:
                api_batches = []
                for batch in batches:
                    messages = [
                        # TODO: Add "Be mindful of idiomatic expressions or cultural references that may require more nuanced translation to convey the correct meaning."?
                        {"role": "user", "content": f"""All messages that I send from now on will contain Japanese. Reply with the translation and only the translation, taking the context of the previous messages into careful consideration.

Additional Meta-data:
```
{metadatas[0]}
```"""},
                        {"role": "assistant", "content": """I understand."""}
                    ]
                    for entry in batch:
                        messages.append({"role": "user", "content": entry.japanese})
                        if entry != batch[-1]:
                            messages.append({"role": "assistant", "content": entry.english})
                            if args.prefill:
                                messages[-1]["content"] = f"{args.prefill} {messages[-1]['content']}"
                    assert messages[-1]["role"] == "user"
                    if args.prefill:
                        messages.append({"role": "assistant", "content": args.prefill})
                    api_batches.append(messages)
                if args.batch_input_file:
                    with open(args.batch_input_file, "a") as f:
                        for messages in api_batches:
                            f.write(f"{json.dumps({'custom_id': f'request-{count}', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': model.model, 'messages': messages, 'max_tokens': MAX_NEW_TOKENS, 'temperature': 0}})}\n")
                    count += 1
                    pbar.update(1)
                    continue
                else:
                    # TODO: Query API
                    print("TODO: Query API")
                    exit(0)
                    results.append(batch_api_output[count])
        elif args.chat_api and isinstance(model, OaiModel):
            for batch in batches:
                messages = [
                    # TODO: Add "Be mindful of idiomatic expressions or cultural references that may require more nuanced translation to convey the correct meaning."?
                    {"role": "user", "content": f"""All messages that I send from now on will contain Japanese. Reply with the translation and only the translation, taking the context of the previous messages into careful consideration.

Additional Meta-data:
```
{metadatas[0]}
```"""},
                    {"role": "assistant", "content": """I understand."""}
                ]
                for entry in batch:
                    messages.append({"role": "user", "content": entry.japanese})
                    if entry != batch[-1]:
                        messages.append({"role": "assistant", "content": entry.english})
                        if args.prefill:
                            messages[-1]["content"] = f"{args.prefill} {messages[-1]['content']}"
                assert messages[-1]["role"] == "user"
                if args.prefill:
                    messages.append({"role": "assistant", "content": args.prefill})
                results.append(model.generate_chat(messages, MAX_NEW_TOKENS, [args.eos_token, "<<JAPANESE>>", *stop_sequences]))
        elif args.batch_size > 1 and model.supports_batching():
            assert len(prompts) == args.batch_size
            results = model.generate_batch(prompts, MAX_NEW_TOKENS, [args.eos_token, "<<JAPANESE>>", *stop_sequences])
        else:
            for prompt in prompts:
                results.append(model.generate(prompt, MAX_NEW_TOKENS, [args.eos_token, "<<JAPANESE>>", *stop_sequences]))

        if not args.batch_input_file and not args.batch_output_file:
            tqdm.write(f"Finished in {time.time()-inf_start}s")

        #print(repr(results))

        for idx, result in enumerate(results):
            if count >= MAX_BENCHMARK_OUTPUT:
                break
            
            prompt = prompts[idx]
            batch = batches[idx]

            expected_english = batch[-1].english

            tqdm.write(f"Japanese: {batch[-1].japanese}")
            tqdm.write(f"Expected ({batch[-1].fidelity}): {expected_english}")
            tqdm.write(f"Generated: {repr(result)}")

            # if not result:
            #     print("Nothing generated, score 0.")
            #     expected_english_full = re.sub(r"【(.*?)】：", r"[\1]: ", expected_english)
            #     scores.append(0)
            #     count += 1
            #     pbar.update(1)
            #     with open(os.path.join(args.results_path, f"{args.title}.jsonl"), "a", encoding="utf-8") as f:
            #         f.write(f"{json.dumps({ 'prompt': prompt, 'expected': expected_english_full+args.eos_token, 'generated': '', 'accuracy': 0.0 })}\n")
            #     continue

            assert result, "Nothing generated."

            result_full = result
            expected_english_full = re.sub(r"【(.*?)】：", r"[\1]: ", expected_english)

            expected_english = expected_english.replace("</s>", "")
            expected_english = expected_english.split("]: ", 1)[-1].split("】：", 1)[-1].split("】:", 1)[-1].split(":** ", 1)[-1]
            
            result = result.replace("</s>", "")
            result = result.split("]: ", 1)[-1].split("】：", 1)[-1].split("】:", 1)[-1].split(":** ", 1)[-1]
            if result.strip():
                result = [s for s in result.split("\n", 1) if s.strip()][0]

            if detect_high_repetition(result):
                tqdm.write("High repetition detected.")
                score = 0
                tqdm.write(f"Score: {score}")
            else:
                tqdm.write("Calculating score...")
                score = get_similarity(expected_english, result)
                tqdm.write(f"Score: {score}")

            scores.append(score)
            tqdm.write(f"ScoreAvg: {sum(scores)/len(scores)}")

            def calculate_stdev(scores):
                scores = [score for score in scores if score > 0]
                return statistics.stdev(scores) if len(scores) > 1 else 0

            def calculate_overall_score(scores, k=1):
                mean = statistics.mean(scores)
                std_dev = calculate_stdev(scores)
                overall_score = mean - k * std_dev
                return overall_score
            
            tqdm.write(f"ScoreAvg DevPen: {calculate_overall_score(scores)}")
            tqdm.write(f"Std Dev: {calculate_stdev(scores)}")

            bleu = get_bleu(expected_english, result) / 100
            chrf = get_chrf(expected_english, result) / 100

            count += 1
            pbar.update(1)
            with open(os.path.join(args.results_path, f"{args.title}.jsonl"), "a", encoding="utf-8") as f:
                f.write(f"{json.dumps({ 'prompt': prompt, 'expected': expected_english_full+args.eos_token, 'generated': result_full+args.eos_token, 'accuracy': float(score), 'bleu': bleu, 'chrf': chrf })}\n")

    pbar.close()

    print(f"All finished. Time Taken: {time.time()-start_bench}s")
