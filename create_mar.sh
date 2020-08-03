#!/bin/bash

torch-model-archiver --model-name chordrecog --version 1.0 --model-file model.py --serialized-file btc_model_large_voca.pt --handler model_handler.py --extra-files run_config.yaml,btc_model_large_voca.pt
