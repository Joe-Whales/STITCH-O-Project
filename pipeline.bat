@REM @echo off
@REM python .\Preprocessing\chunker.py preprocess_config-1.yaml
@REM python .\Preprocessing\calc_stats.py --separate chunks/ 1
@REM python .\Preprocessing\train_test_split.py chunks/
@REM python .\Preprocessing\scale_data_multi.py
@REM python .\Preprocessing\generate_metadata.py
@REM python .\UniAD\train_val.py --config train_config.yaml
@REM pause
@echo off
python .\Preprocessing\chunker.py preprocess_config-1.yaml
python ./Preprocessing/process_chunks.py chunks chunks_scaled
python .\Preprocessing\generate_metadata.py
@REM python .\UniAD\train_val.py --config train_config.yaml
pause