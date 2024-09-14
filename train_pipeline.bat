@echo off
python .\Preprocessing\chunker.py preprocess_config-big.yaml
python ./Preprocessing/process_chunks.py chunks chunks_scaled
python .\Preprocessing\generate_metadata.py chunks_scaled -t
python .\UniAD\train_val.py --config train_config.yaml
pause 