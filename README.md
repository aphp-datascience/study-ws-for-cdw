# Source code for the submitted article "From clinical narratives to interoperable real-world data: weak supervision for clinical data warehouse enrichment"


# Installation
## Environment
```bash
cd wedsak
python -m venv .venv
source .venv/bin/activate
uv sync --active
```

### Make a specific env for vllm
```bash
# Dans jupyter
conda create -n py312 python=3.12.9
conda activate py312
python -m venv ./venvs/.venv_vllm
source venvs/.venv_vllm/bin/activate
pip install -r venvs/requirements_vllm.txt
```

## Make a kernel
```bash
python -m ipykernel install --user --name=wedsak_kernel
eds-toolbox kernel --spark
```


## Activate environment
```bash
cd wedsak
source .venv/bin/activate
```

## Launch slurm kernel
```bash
slurm-kernel launch a100
```


# LLM - how to? 
## Launch slurm kernel
```bash
source ./wedsak/.venv/bin/activate
slurm-kernel launch a100 --node a100_bbs-edsg28-p016 --connect
```

## Connect ssh to gpu
```bash
ssh a100_bbs-edsg28-p017 # ssh a100_bbs-edsg28-p016 # ssh h100_bbs-edsgpu-p019 # ssh h100_bbs-edsgpu-p020
source ./venvs/.venv_vllm/bin/activate
conda deactivate
watch -n0.1 nvidia-smi
```

### Workflow to launch a vLLM server
```bash
source ./wedsak/.venv/bin/activate
slurm-kernel launch a100 --node a100_bbs-edsg28-p017 --connect
source ./venvs/.venv_vllm/bin/activate
conda deactivate

CUDA_VISIBLE_DEVICES=0 HF_HOME="/data/hdd/cse250022/hf_cache/" vllm serve google/medgemma-27b-text-it --port 8003 --enable-prefix-caching  --download-dir /data/hdd/cse250022/hf_cache/hub --host 0.0.0.0 --max-model-len 90400 --data-parallel-size 1 --tensor-parallel-size 1

CUDA_VISIBLE_DEVICES=1 HF_HOME="/data/hdd/cse250022/hf_cache/" vllm serve Qwen/Qwen3-8B --port 8002 --enable-prefix-caching --max-num-batched-tokens=32768 --download-dir /data/hdd/cse250022/hf_cache/hub --reasoning-parser deepseek_r1 --max-model-len 32768 --host 0.0.0.0 --data-parallel-size 1 --tensor-parallel-size 1
```


# Scripts

```bash
init_wedsak


python scripts/retrieve_external_knowledge.py --config config/conf_general.yml --config config/conf_train.yml
python scripts/apply_no_llm_lf.py --config config/conf_general.yml --config config/conf_train.yml
python scripts/apply_llm_lf.py --config config/conf_general.yml --config config/conf_train.yml 
python scripts/merge_lf_results.py --config config/conf_general.yml --config config/conf_train.yml
python scripts/label_modelling.py --config config/conf_general.yml --config config/conf_train.yml
python scripts/data_selection.py --config config/conf_general.yml --config config/conf_train.yml
# one dataset all tasks
python scripts/train2.py --config config/conf_general.yml --config config/conf_train.yml
# multiple specific task datasets
python scripts/train.py --config config/conf_general.yml --config config/conf_train.yml

# Evaluate
python scripts/evaluate.py --model-path '/export/home/cse250022/wedsak/data/models/all_tasks_no_dropout_more_context' --dataset-path '/export/home/cse250022/wedsak/data/annotation/test/docs_group_B_0_49_AC'

python scripts/evaluate.py --config config/conf_test.yml  

# Evaluate all models (batch)
python scripts/evaluate_all.py --path-model-refs '/export/home/cse250022/wedsak/data/model_refs.xlsx' --path-dataset-A '/export/home/cse250022/wedsak/data/annotation/test/docs_group_A' --path-dataset-B '/export/home/cse250022/wedsak/data/annotation/test/docs_group_B' --types-to-evaluate "['rule_based', 'llm', 'WS']" --name-code-filter "['all_tasks_hlm']"

# Inference
python scripts/inference.py --model-path '/export/home/cse250022/wedsak/data/models/all_tasks' --data-path '~/wedsak/data/annotation/test/test_set_B.csv' --output-path '~/wedsak/data/annotation/test/test_set_B_inference_model_all_tasks.parquet'

python scripts/inference.py --model-path '/export/home/cse250022/wedsak/data/models/all_tasks' --data-path '~/wedsak/data/annotation/test/test_set_A.csv' --output-path '~/wedsak/data/annotation/test/test_set_A_inference_model_all_tasks.parquet'

## Task independent
python scripts/label_modelling.py --config config/conf_general.yml --config config/conf_train.yml --task-ids [16]
python scripts/data_selection.py --config config/conf_general.yml --config config/conf_train.yml --config config/conf_task_independent.yml --task-ids [2]
