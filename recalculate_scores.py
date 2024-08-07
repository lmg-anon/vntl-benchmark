import os
import json
import sacrebleu
import re
from tqdm import tqdm

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

def clean_text(text, stricter=False):
    if stricter:
        text = re.sub(r"([^a-zA-Z]|^)([a-zA-Z])(?i:-\2)+([a-zA-Z])", r"\1\2\3", text)
    to_strip = "&っ。～―（）「」｢｣『』“”\"'，、○()«»~ \t\r\n"
    if stricter:
        to_strip += "….?？!！,"
    text = text.strip(to_strip)
    return text

def get_similarity(ref, hyp):
    ref = clean_text(ref, stricter=True)
    if not ref:
        return 1.0
    hyp = clean_text(hyp, stricter=True)
    if ref.lower() == hyp.lower():
        return 1.0
    return float(get_similarity_batched([ref], [hyp])[0])

def get_bleu(ref, hyp):
    ref = clean_text(ref)
    hyp = clean_text(hyp)
    if ref.lower() == hyp.lower():
        return 100
    bleu = sacrebleu.sentence_bleu(hyp, [ref])
    return bleu.score

def get_chrf(ref, hyp):
    ref = clean_text(ref)
    hyp = clean_text(hyp)
    if ref.lower() == hyp.lower():
        return 100
    chrf = sacrebleu.sentence_chrf(hyp, [ref])
    return chrf.score

def process_file(file_path):
    with open(file_path, 'r') as file:
        try:
            os.remove(file_path.replace("results", "results_new"))
        except:
            pass    
        for line in tqdm(file, total=128):
            entry = json.loads(line)

            source = entry["prompt"].replace("</s>", "")
            if "<<JAPANESE>>" in entry["prompt"]:
                source = source[source.rindex("<<JAPANESE>>"):]
            source = source.split("]: ", 1)[-1].split("】：", 1)[-1].split("】:", 1)[-1]
            source = source.split("<<ENGLISH>>", 1)[0].strip()

            expected = entry["expected"].replace("</s>", "")
            expected = expected.split("]: ", 1)[-1].split("】：", 1)[-1].split("】:", 1)[-1]
            
            generated = entry["generated"].replace("</s>", "")
            generated = generated.split("]: ", 1)[-1].split("】：", 1)[-1].split("】:", 1)[-1]
            if generated.strip():
                generated = [s for s in generated.split("\n", 1) if s.strip()][0]

            entry["accuracy"] = get_similarity(expected, generated)

            entry["bleu"] = get_bleu(expected, generated) / 100
            entry["chrf"] = get_chrf(expected, generated) / 100

            with open(file_path.replace("results", "results_new"), 'a') as file:
                file.write(json.dumps(entry)+"\n")

input_folder = "results"

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.jsonl') and not filename.startswith('_'):
        file_path = os.path.join(input_folder, filename)
        process_file(file_path)