"""Compute ROUGE-L, BERTScore, and sentence-similarity metrics."""

import os
from rouge_chinese import Rouge
import jieba
import contextlib
import io
import logging
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Reduce third-party progress and logging noise.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    from transformers import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
    if hasattr(hf_logging, "disable_progress_bar"):
        hf_logging.disable_progress_bar()
except Exception:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)


@contextlib.contextmanager
def _silence_stdout_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# Configure portable input and output roots through environment variables.
input_root_path = os.environ.get('CLEANED_ROOT', './test_results_diagnosis')
output_root_path = os.environ.get('SCORES_ROOT', './test_results_scores')

# Model result directories included in the paper comparison.
file_path_list = [
    'finetuning/deepseek-r1-qwen3-8b',
    'finetuning_sr/deepseek-r1-qwen3-8b',
    # 'finetuning_sr_ICL/deepseek-r1-qwen3-8b',
    # 'finetuning_kg/deepseek-r1-qwen3-8b',
    # 'finetuning_kg_v2/deepseek-r1-qwen3-8b',
    'finetuning_kg_v3/deepseek-r1-qwen3-8b',
    # 'finetuning_kg_v3_v2/deepseek-r1-qwen3-8b',
    # 'finetuning_kg_v3_ICL/deepseek-r1-qwen3-8b',
    'finetuning_DAPT/deepseek-r1-qwen3-8b',
    'finetuning_ICL/deepseek-r1-qwen3-8b',
    # 'finetuning_combined_ICL_woICL/deepseek-r1-qwen3-8b',
    'finetuning_combined_ICL_ICL/deepseek-r1-qwen3-8b',
    'FetalScribe/deepseek-r1-qwen3-8b',

    'with_ICL/closed_source/gpt-5.1',
    'with_ICL/closed_source/gpt-5.2',
    'with_ICL/closed_source/gpt-5.4',
    'with_ICL/closed_source/gemini-3.1-flash-lite-preview',
    'with_ICL/closed_source/gemini-3.1-pro-preview',
    'with_ICL/closed_source/claude-sonnet-4-6',
    'with_ICL/closed_source/claude-opus-4-6',
    'with_ICL/closed_source/claude-opus-4-7',
    'with_ICL/closed_source/grok-4.1',
    'with_ICL/closed_source/grok-4.2-nothinking',
    'with_ICL/closed_source/doubao-seed-2-0-lite-260215',
    'with_ICL/closed_source/qwen3-max',
    'with_ICL/open_source/qwen3-235b-a22b',
    'with_ICL/open_source/DeepSeek-V3.2',
    # 'with_ICL/open_source/MiMo-V2-Flash',
    'with_ICL/open_source/kimi-k2.5',
    'with_ICL/open_source/glm-5',
    'with_ICL/AntAngelMed', 
    'with_ICL/HuatuoGPT-Vision-34B',
    'with_ICL/OpenBioLLM-70B-4bit',

    'without_ICL/closed_source/gpt-5.1',
    'without_ICL/closed_source/gpt-5.2',
    'without_ICL/closed_source/gpt-5.4',
    'without_ICL/closed_source/gemini-3.1-flash-lite-preview',
    'without_ICL/closed_source/gemini-3.1-pro-preview',
    'without_ICL/closed_source/claude-sonnet-4-6',
    'without_ICL/closed_source/claude-opus-4-6',
    'without_ICL/closed_source/claude-opus-4-7',
    'without_ICL/closed_source/grok-4.1',
    'without_ICL/closed_source/grok-4.2-nothinking',
    'without_ICL/closed_source/doubao-seed-2-0-lite-260215',
    'without_ICL/closed_source/qwen3-max',
    'without_ICL/open_source/qwen3-235b-a22b',
    'without_ICL/open_source/DeepSeek-V3.2',
    # 'without_ICL/open_source/MiMo-V2-Flash',
    'without_ICL/open_source/kimi-k2.5',
    'without_ICL/open_source/glm-5',
    'without_ICL/AntAngelMed', 
    'without_ICL/HuatuoGPT-Vision-34B',
    'without_ICL/OpenBioLLM-70B-4bit'
]

# Evaluation datasets.
dataset_list = [
    'InDistribution_test.json',
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json'
]


def getRougeLScore(output, label):
    try:
        output_terms = ' '.join(jieba.cut(output))
        label_terms = ' '.join(jieba.cut(label))
        rouge_scores = Rouge().get_scores(output_terms, label_terms)
        rouge_scores = rouge_scores[0]['rouge-l']
        return rouge_scores['f'], rouge_scores['p'], rouge_scores['r']
    except:
        return 0, 0, 0


def getBERTScore_with_scorer(output, label, scorer: BERTScorer):
    try:
        p, r, f1 = scorer.score([output], [label])
        return f1.mean().item(), p.mean().item(), r.mean().item()
    except Exception:
        return 0, 0, 0


def getSentenceSimilarity(output, label, model=None):
    try:
        output_embedding = model.encode([output])
        label_embedding = model.encode([label])
        sentence_similarity = util.pytorch_cos_sim(output_embedding, label_embedding)
        if isinstance(sentence_similarity, list):
            sentence_similarity = sentence_similarity[0]
        if isinstance(sentence_similarity, list):
            sentence_similarity = sentence_similarity[0]
        return sentence_similarity.item()
    except:
        return 0


def load_file_and_process(full_name, sentence_model, bert_scorer: BERTScorer):
    try:
        with open(full_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert isinstance(data, list), 'not supported data!'

        predicted_scores = []

        for idx, patient in enumerate(data):
            # Use the cleaned fields produced by stage 1.
            cleaned_output = patient.get('cleaned_output', '')
            cleaned_label  = patient.get('cleaned_label',  '')

            rouge_f1, rouge_p, rouge_r = getRougeLScore(cleaned_output, cleaned_label)
            bert_f1, bert_p, bert_r   = getBERTScore_with_scorer(cleaned_output, cleaned_label, scorer=bert_scorer)
            sentence_sim              = getSentenceSimilarity(cleaned_output, cleaned_label, model=sentence_model)

            predicted_scores.append({
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
                print(f'  已处理 {idx + 1}/{len(data)} 条...')

        return predicted_scores

    except FileNotFoundError:
        print(f'错误: 找不到文件 {full_name}')
        return None


if __name__ == '__main__':

    # Load shared embedding models once for all datasets.
    print('正在加载评估模型（SentenceTransformer + BERTScore）...')
    with _silence_stdout_stderr(enabled=True):
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        bert_scorer = BERTScorer(lang="zh", model_type="bert-base-chinese")
    print('模型加载完毕!')

    for model_name in file_path_list:
        for dataset_name in dataset_list:
            output_dir = os.path.join(output_root_path, model_name)
            os.makedirs(output_dir, exist_ok=True)

            scores_save_path = os.path.join(output_dir, '[Scores]_' + dataset_name.replace('.json', '.csv'))
            if os.path.exists(scores_save_path):
                print(f'  已存在结果文件，跳过: {scores_save_path}')
                continue 

            cleaned_dataset_name = '[Cleaned]_' + dataset_name
            file_full_name = os.path.join(input_root_path, model_name, cleaned_dataset_name)

            print(f'\n开始评估: 模型={model_name} | 数据={cleaned_dataset_name}')
            predicted_scores = load_file_and_process(file_full_name, sentence_model, bert_scorer)

            if predicted_scores is None:
                print(f'  跳过（文件不存在）')
                continue

            scores_df = pd.DataFrame(predicted_scores)
            scores_df.to_csv(scores_save_path, index=False, encoding='utf-8-sig')

            print(f'  评估完成，各指标均值：')
            for col in scores_df.columns:
                if col != 'patient_id':
                    print(f'    {col}: {scores_df[col].mean():.4f}')
            print(f'  结果已保存: {scores_save_path}')
