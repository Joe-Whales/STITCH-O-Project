@echo off

python sam-segmentation/segmentation.py sam-segmentation/segmentation-config.yaml Preprocessing/data/
python .\Preprocessing\chunker.py preprocess_config_inference.yaml
python ./Preprocessing/process_chunks.py chunks chunks_inference
python .\Preprocessing\generate_metadata.py chunks_inference -t
python .\UniAD\run_inference.py --config inference_config.yaml

pause
