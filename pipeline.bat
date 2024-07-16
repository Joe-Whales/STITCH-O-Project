@echo off
python .\Preprocessing\chunker.py preprocess_config-1.yaml
python .\Preprocessing\calc_stats.py --separate chunks/ 1
python .\Preprocessing\train_test_split.py chunks/
python .\Preprocessing\scale_data_multi.py
python .\Preprocessing\generate_metadata.py
python .\UniAD\train_val.py --config train_config.yaml
pause