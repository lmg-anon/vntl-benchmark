import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings

from colorama import Fore, init as colorama_init
from modules.model import LanguageModel
from modules.model.backends import KcppModel, LcppModel, OobaModel, OaiModel, LPY_PRESENT, EXL2_PRESENT, TF_PRESENT, UNSLOTH_PRESENT
if LPY_PRESENT:
    from modules.model.backends import LpyModel
if EXL2_PRESENT:
    from modules.model.backends import EXL2Model
if TF_PRESENT:
    from modules.model.backends import TFModel
if UNSLOTH_PRESENT:
    from modules.model.backends import UnslothModel
from modules.visual_novel import *
from modules.translation import *
from modules.log import Logger
from tqdm import tqdm
import torch
import argparse
import json
import time
import os
import re
import random


st_model = None
def get_similarity_batched(texts1, texts2):
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
    if text1.strip().lower() == text2.strip().lower():
        return 1.0
    return float(get_similarity_batched([text1], [text2])[0])

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

def compile_metadata(characters):
    metadata_sections = []
    for character in characters:
        metadata_section = f"Name: {character.name} ({character.japanese_name}) | Gender: {character.gender}"
        if character.aliases and character.aliases != "None":
            metadata_section += f" | Aliases: {character.aliases}"
        metadata_sections.append(metadata_section)
    return '<<START>>\n' + '\n'.join(random.sample(metadata_sections, k=len(metadata_sections)))

if __name__ == "__main__":
    MAX_ENTRIES_PER_BATCH = 10
    MAX_NEW_TOKENS = 120
    QUALITY_THRESHOLD = 1 # 0.83
    MAX_BENCHMARK_OUTPUT = 128

    SUPPORTED_BACKENDS = {"koboldcpp", "llamacpp", "ooba", "llamapy", "openai", "exl2", "transformers", "unsloth"}
    if not LPY_PRESENT:
        SUPPORTED_BACKENDS.remove("llamapy")
    if not EXL2_PRESENT:
        SUPPORTED_BACKENDS.remove("exl2")
    if not TF_PRESENT:
        SUPPORTED_BACKENDS.remove("transformers")
    if not UNSLOTH_PRESENT:
        SUPPORTED_BACKENDS.remove("unsloth")

    parser = argparse.ArgumentParser(description="VNTL Benchmark")

    parser.add_argument("--title", type=str, help="run title")

    parser.add_argument("--backend", type=str, choices=SUPPORTED_BACKENDS, help="model backend type")
    parser.add_argument("--preset", type=str, help="model preset (default: default)")
    parser.add_argument("--context-size", type=int, help="model context size (default: 2048)", default=2048)
    parser.add_argument("--batch-size", type=int, help="model batch size (default: 1)", default=1)
    #parser.add_argument("--format", type=str, help="model prompt format (default: alpaca)")
    parser.add_argument("--model", type=str, help="model path for non-api backends")
    parser.add_argument("--host", type=str, help="host for the model backend")
    parser.add_argument("--api-key", type=str, help="api key for the model backend")

    parser.add_argument("--seed", type=int, help="initial rng seed")

    parser.add_argument("--verbose", action="store_true", help="enable verbose output")

    args = parser.parse_args()

    print(args.title)

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    colorama_init()
    Logger.init()

    if args.seed:
        random.seed(args.seed)
        LanguageModel.base_seed = args.seed

    if args.verbose:
        Logger.print_verbose = True

    scores = []
    processed_prompts = []
    if os.path.isfile(f"./results/{args.title}.jsonl"):
        while True:
            reply = input("Do you want to resume the previous run? (Y/N): ").strip().lower()
            if reply == "y":
                with open(f"./results/{args.title}.jsonl", "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        obj = json.loads(line)
                        scores.append(obj["score"])
                        processed_prompts.append(obj["prompt"])
                break
            elif reply == "n":
                try:
                    os.remove(f"./results/{args.title}.jsonl")
                except:
                    pass
                break
    
    model = OaiModel(args.host, args.api_key, args.context_size)
    model.wait()

    EOS_TOKEN = "</s>"

    model.presets = {
        "temperature": 0,
        "repetition_penalty": 1,
    }

    # Parse the characters file
    books = VisualNovels("characters.txt")
    books.read_file()

    start_bench = time.time()

    # 'TLAssist-test-val.txt'
    # 'TLAssist-validation-v4_SIMILARITY_NS.txt'
    # 'TLAssist-all-v4_SIMILARITY_NS.txt'
    translation_file_path = 'TLAssist-validation-v4_SIMILARITY_NS.txt'
    tf = TranslationFile(translation_file_path)
    tf.read_file()

    entries = tf.entries
    last_entry = None

    skipping = len(processed_prompts) > 0

    entry_batches: list[list[TranslationEntry]] = [[]]
    prompt_batches: list[list[str]] = [[]]
    for idx in range(len(entries)):
        batch = entries[max(idx-MAX_ENTRIES_PER_BATCH-1, 0):idx+1]
        if last_entry == batch[-1]:
            break
        last_entry = batch[-1]
        batch = [entry for entry in batch if entry.book_id == last_entry.book_id]

        if "absolute" not in batch[-1].fidelity and "high" not in batch[-1].fidelity:
            continue

        # We want the LLM to have some completion examples.
        if idx < 3:
            continue

        ID = batch[-1].book_id

        block = ""
        for entry in batch:
            block += \
                f"<<JAPANESE>>\n" \
                f"{entry.japanese}\n" \
                f"<<ENGLISH>>\n" \
                f"{entry.english if entry != batch[-1] else ''}{EOS_TOKEN if entry != batch[-1] else ''}\n"
        block = block.strip() + "\n"
        block = re.sub(r"【(.*?)】：", r"[\1]: ", block)

        # Find all unique speaking characters in this block
        speaking_characters: set[Character] = set()
        speaking_names = set(re.findall(r'\[(.*?)\]', block))
        for name in speaking_names:
            _character = books.get_character(ID, name)
            if _character:
                speaking_characters.add(_character)
        metadata = compile_metadata(speaking_characters)
        prompt = metadata + "\n" + block
        
        #prompt = f"[INST] All messages that I will send from now on will contain Japanese, reply with the translation and only the translation, taking the context of the previous messages into careful consideration.\nAdditional Meta-data:\n```\n{metadata}\n```\nNote: Always start your replies with \"<<ENGLISH>>\". [/INST] Sure thing! Please send the Japanese!</s>" + block

        if skipping:
            skipping = processed_prompts[-1][processed_prompts[-1].index("<<JAPANESE>>"):] != prompt[prompt.index("<<JAPANESE>>"):]
            continue

        entry_batches[-1].append(batch)
        prompt_batches[-1].append(prompt)
        if len(entry_batches[-1]) == args.batch_size:
            entry_batches.append([])
            prompt_batches.append([])

    count = len(processed_prompts)
    pbar = tqdm(total=MAX_BENCHMARK_OUTPUT-len(processed_prompts))
    for batches, prompts in zip(entry_batches, prompt_batches):
        if count >= MAX_BENCHMARK_OUTPUT:
            break

        tqdm.write("")
        tqdm.write("Generating...")

        inf_start = time.time()

        if args.batch_size > 1 and model.supports_batching():
            results = model.generate_batch(prompts, args.batch_size, MAX_NEW_TOKENS)
        else:
            results = []
            for prompt in prompts:
                results.append(model.generate(prompt, MAX_NEW_TOKENS, [EOS_TOKEN, "<<JAPANESE>>"]))

        tqdm.write(f"Finished in {time.time()-inf_start}s")

        #print(repr(results))

        for idx, result in enumerate(results):
            if count >= MAX_BENCHMARK_OUTPUT:
                break
            
            prompt = prompts[idx]
            batch = batches[idx]

            expected_english = batch[-1].english

            #tqdm.write(f"Japanese: {batch[-1].japanese}")
            #tqdm.write(f"Expected ({batch[-1].fidelity}): {expected_english}")
            #tqdm.write(f"Generated: {result}")
            assert result, repr(result)

            result_full = result
            expected_english_full = re.sub(r"【(.*?)】：", r"[\1]: ", expected_english)

            expect_split = False
            if "】：" in expected_english:
                expect_split = True
                expected_english = expected_english.split("】：")[1]
            elif "]: " in expected_english:
                expect_split = True
                expected_english = expected_english.split("]: ")[1]
            if "】：" in result:
                expect_split = False
                result = result.split("】：")[1]
            elif "]: " in result:
                expect_split = False
                result = result.split("]: ")[1]
            #assert not expect_split, (repr(prompt), repr(result))

            tqdm.write("Calculating score...")
            score = get_similarity(expected_english, result)
            tqdm.write(f"Score: {score}")

            scores.append(score)
            tqdm.write(f"ScoreAvg: {sum(scores)/len(scores)}")

            if score < QUALITY_THRESHOLD and not detect_high_repetition(result):
                count += 1
                pbar.update(1)
                with open(f"./results/{args.title}.jsonl", "a", encoding="utf-8") as f:
                    f.write(f"{json.dumps({ 'prompt': prompt, 'chosen': expected_english_full+EOS_TOKEN, 'rejected': result_full+EOS_TOKEN, 'score': float(score) })}\n")

    pbar.close()

    print(f"All finished. Time Taken: {time.time()-start_bench}s")
