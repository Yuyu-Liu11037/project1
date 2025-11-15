#!/bin/bash

python mimic_iv.py --model_type transformer --train_percentage 1.0 --lr 2e-5 \
    > logs/transformer_$(date +%Y%m%d_%H%M%S).log 2>&1

# python mimic_iv.py --model_type svm --train_percentage 1.0 \
#     > logs/svm_$(date +%Y%m%d_%H%M%S).log 2>&1

python mimic_iv.py --model_type transformer --train_percentage 1.0 --use_hyperbolic_embeddings --lr 2e-5 \
    > logs/transformer_hyperbolic_$(date +%Y%m%d_%H%M%S).log 2>&1

# python mimic_iv.py --model_type svm --train_percentage 1.0 --use_hyperbolic_embeddings \
#     > logs/svm_hyperbolic_$(date +%Y%m%d_%H%M%S).log 2>&1
