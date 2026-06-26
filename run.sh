set +e

CONFIG=$1
CONFIG_GENERAL=config/conf_general.yml

python scripts/retrieve_external_knowledge.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/apply_no_llm_lf.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/apply_llm_lf.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/merge_lf_results.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/label_modelling.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/data_selection.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/train.py --config $CONFIG_GENERAL --config $CONFIG
python scripts/train2.py --config $CONFIG_GENERAL --config $CONFIG