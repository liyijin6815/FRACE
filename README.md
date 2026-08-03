# FetalScribe: Fetal Brain MRI Report Generation

This repository contains the training, knowledge-graph augmentation, domain-adaptive pre-training, inference, and evaluation code used for fetal brain MRI diagnostic report generation.

The base model is [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B). The released model files are parameter-efficient LoRA adapters and must be loaded together with the base model.

![Method overview](fig1.jpg)

## Repository layout

```text
.
|-- data_examples/
|   `-- example_dataset.json
|-- Finetuning/
|   |-- configs/
|   |-- data/dataset_info.json
|   `-- adapter/fetal_mri_lora/
|-- Finetuning_KG/
|   |-- configs/
|   |-- data/dataset_info.json
|   |-- kg_rag/
|   |-- table/
|   |-- tests/
|   `-- adapter/fetal_mri_plus_kg_lora_v3/
|-- Finetuning_DAPT/
|   |-- configs/
|   |-- scripts/
|   `-- adapters/
|       |-- direct/{center1,center2}/
|       `-- kg/{center1,center2}/
`-- evaluation/
```

Patient reports are private and are not included. `data_examples/example_dataset.json` contains one fully synthetic record that documents the expected schema.

## Installation

The experiments used the `lyj_llamafactory` environment with Python 3.12.13, PyTorch 2.5.1+cu121, Transformers 5.2.0, PEFT 0.18.1, and LLaMA-Factory 0.9.5.dev0. Install the CUDA 12.1 PyTorch build, this repository's dependencies, and the exact LLaMA-Factory revision:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 833f6027b17a5502664371f901827844b9fad6fa
pip install -e ".[torch,metrics]"
```

All commands below are run from the repository root unless stated otherwise. Private datasets must use the four fields shown in `data_examples/example_dataset.json`: `instruction`, `input`, `output`, and `system`.

## 1. Direct supervised fine-tuning

Place the private JSON files referenced by `Finetuning/data/dataset_info.json` in `Finetuning/data/`.

```bash
cd Finetuning
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/train.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/predict.yaml
```

The released direct-SFT adapter is in `Finetuning/adapter/fetal_mri_lora/`.

## 2. Knowledge-graph augmented fine-tuning

The KG pipeline has two retrieval paths:

1. Numeric measurements are extracted at clause level, normalized to centimeters, paired with the correct anatomy and laterality, and graded using KG thresholds.
2. Qualitative findings are matched through aliases with local negation filtering, followed by typed graph traversal through `Indicates` and `AssocWith` relations.

The retrieved evidence is verbalized and appended to the report under `Clinical Knowledge Reference`. The model architecture is unchanged, but KG retrieval and context construction are required during both training and inference.

Build a KG-augmented JSON file:

```bash
python -m Finetuning_KG.kg_rag.run_build \
  --input-path ./private_data/input.json \
  --output-path ./private_data/augmented.json \
  --kg-path Finetuning_KG/kg_rag/clinical_kg.json \
  --stats
```

Place the augmented private files referenced by `Finetuning_KG/data/dataset_info.json` in `Finetuning_KG/data/`, then run:

```bash
cd Finetuning_KG
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/train_kg.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/predict_kg.yaml
```

`Finetuning_KG/table/` contains bilingual entity tables and reproducible table-generation scripts. The regression tests use synthetic sentences only:

```bash
python Finetuning_KG/table/generate_entity_tables.py \
  --kg-path Finetuning_KG/kg_rag/clinical_kg.json \
  --language cn

python Finetuning_KG/table/generate_entity_tables.py \
  --kg-path Finetuning_KG/table/clinical_kg_en.json \
  --language en

python -m unittest Finetuning_KG.tests.test_kg_rag
```

## 3. Domain-adaptive pre-training

DAPT uses only the unlabeled `input` field from each target center. A small source-domain replay sample is mixed into the target corpus to reduce catastrophic forgetting. A separate DAPT adapter is trained for each center and each SFT branch.

```bash
python Finetuning_DAPT/scripts/01_extract_unlabeled_text.py \
  --dataset-dir ./private_data \
  --output-dir Finetuning_DAPT/data

python Finetuning_DAPT/scripts/02_build_dapt_corpus.py \
  --data-dir Finetuning_DAPT/data \
  --replay 0.006

bash Finetuning_DAPT/scripts/03_train_dapt.sh all
bash Finetuning_DAPT/scripts/04_predict.sh all all
```

The loading chain is:

```text
base model + matching SFT adapter + center-specific DAPT adapter
```

For example, center-1 KG+DAPT inference loads:

```text
DeepSeek-R1-0528-Qwen3-8B
  + Finetuning_KG/adapter/fetal_mri_plus_kg_lora_v3
  + Finetuning_DAPT/adapters/kg/center1
```

The four DAPT adapters correspond to direct SFT and KG-augmented SFT for centers 1 and 2. A DAPT adapter is not a standalone model.

## Adapter files and reproducibility

Each released adapter directory contains:

- `adapter_model.safetensors`: learned LoRA parameters.
- `adapter_config.json`: LoRA architecture, target modules, rank, scaling, dropout, and base-model metadata.

These two files are sufficient to load an adapter for inference. Reproducing training additionally requires the base model, private training data, the corresponding YAML configuration, preprocessing code, software environment, and random seed. Optimizer states, scheduler states, trainer logs, and intermediate checkpoints are not required for inference and are intentionally excluded.

## Evaluation

The evaluation pipeline contains:

1. Prediction extraction and text cleaning.
2. ROUGE-L, BERTScore, and sentence-similarity computation.
3. GPT-5.2 clinical-keyword extraction.
4. GPT-5.2 semantic matching and TP/FP/FN counting.
5. Bootstrap confidence intervals and paired significance tests.

The statistical analysis uses 1,000 bootstrap resamples, one-sided paired Wilcoxon signed-rank tests for NLP metrics, and 10,000 paired permutations for clinical metrics.

Set API credentials through environment variables. Never place credentials in source files:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export EVAL_MODEL="gpt-5.2"
```

The evaluation scripts use relative result roots that can be overridden with `PREDICTIONS_ROOT`, `CLEANED_ROOT`, `SCORES_ROOT`, `LLM_ROOT`, and `METRICS_ROOT`. The expected files and exact execution order are documented in `evaluation/README.md`.

## Data privacy

The fetal MRI reports cannot be released because of patient privacy and institutional restrictions. No patient data, generated patient reports, API credentials, local absolute paths, optimizer states, or caches are included in this repository.

## Citation

Citation metadata will be added after publication.

## License

The code and released research artifacts are provided for non-commercial academic research. See `LICENSE`.
