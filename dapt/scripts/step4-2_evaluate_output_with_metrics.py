"""
NLP metrics evaluation: computes ROUGE-L, BERTScore, and Sentence Similarity.

Usage:
  export HF_ENDPOINT=https://hf-mirror.com  # optional mirror
  DIAG_ROOT=./results/diagnosis SCORES_ROOT=./results/scores MODEL_NAME=model \
    CUDA_VISIBLE_DEVICES=0 python step4-2_evaluate_output_with_metrics.py
"""

import os
import json
import warnings
import logging
import contextlib
import io
import pandas as pd
warnings.filterwarnings('ignore')

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    if hasattr(hf_logging, "disable_progress_bar"):
        hf_logging.disable_progress_bar()
except Exception:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)

from rouge_chinese import Rouge
import jieba
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util


@contextlib.contextmanager
def _silence(enabled=True):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


input_root_path  = os.environ.get('DIAG_ROOT', './results/diagnosis')
output_root_path = os.environ.get('SCORES_ROOT', './results/scores')
file_path_list = [os.environ.get('MODEL_NAME', 'model')]

dataset_list = [
    'InDistribution_test.json',
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json',
]


def getRougeLScore(output, label):
    try:
        output_terms = ' '.join(jieba.cut(output))
        label_terms  = ' '.join(jieba.cut(label))
        scores = Rouge().get_scores(output_terms, label_terms)[0]['rouge-l']
        return scores['f'], scores['p'], scores['r']
    except:
        return 0, 0, 0


def getBERTScore(output, label, scorer):
    try:
        p, r, f1 = scorer.score([output], [label])
        return f1.mean().item(), p.mean().item(), r.mean().item()
    except:
        return 0, 0, 0


def getSentenceSimilarity(output, label, model):
    try:
        sim = util.pytorch_cos_sim(model.encode([output]), model.encode([label]))
        return sim.item()
    except:
        return 0


def load_file_and_process(full_name, sentence_model, bert_scorer):
    with open(full_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores = []
    for idx, patient in enumerate(data):
        cleaned_output = patient.get('cleaned_output', '')
        cleaned_label  = patient.get('cleaned_label',  '')

        rouge_f1, rouge_p, rouge_r = getRougeLScore(cleaned_output, cleaned_label)
        bert_f1,  bert_p,  bert_r  = getBERTScore(cleaned_output, cleaned_label, bert_scorer)
        sentence_sim               = getSentenceSimilarity(cleaned_output, cleaned_label, sentence_model)

        scores.append({
            'patient_id':   patient.get('patient_id', idx),
            'rouge_f1':     rouge_f1,
            'rouge_p':      rouge_p,
            'rouge_r':      rouge_r,
            'bert_f1':      bert_f1,
            'bert_p':       bert_p,
            'bert_r':       bert_r,
            'sentence_sim': sentence_sim,
        })

        if (idx + 1) % 10 == 0:
            print(f'  Processed {idx + 1}/{len(data)}...')

    return scores


if __name__ == '__main__':
    print('Loading evaluation models (SentenceTransformer + BERTScore)...')
    with _silence(enabled=True):
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        bert_scorer    = BERTScorer(lang="zh", model_type="bert-base-chinese")
    print('Models loaded!')

    for model_name in file_path_list:
        for dataset_name in dataset_list:
            output_dir = os.path.join(output_root_path, model_name)
            os.makedirs(output_dir, exist_ok=True)

            scores_save_path = os.path.join(output_dir, f'[Scores]_{dataset_name.replace(".json", "")}.csv')
            if os.path.exists(scores_save_path):
                print(f'  Already exists, skip: {scores_save_path}')
                continue

            file_full_name = os.path.join(input_root_path, model_name, f'[Cleaned]_{dataset_name}')
            if not os.path.exists(file_full_name):
                print(f'  File not found, skip: {file_full_name}')
                continue

            print(f'\nEvaluating: {model_name} | {dataset_name}')
            predicted_scores = load_file_and_process(file_full_name, sentence_model, bert_scorer)

            df = pd.DataFrame(predicted_scores)
            df.to_csv(scores_save_path, index=False, encoding='utf-8-sig')

            print(f'  Evaluation done, mean scores:')
            for col in df.columns:
                if col != 'patient_id':
                    print(f'    {col}: {df[col].mean():.4f}')
            print(f'  Results saved: {scores_save_path}')
