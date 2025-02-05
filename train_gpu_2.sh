#!/bin/bash

python train.py \
    --device 2 \
    --phase train \
    #--resume ./trained_models/text_unet/best_metric_model.pth \
    --model_name text_unet \
    --cache_dataset \
    --dataset_json ./dataset/dataset_json_files/ballanced.json