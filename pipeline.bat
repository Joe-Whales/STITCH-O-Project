@echo off
python chunker.py config.yaml
python train_test_split.py chunks/
pause