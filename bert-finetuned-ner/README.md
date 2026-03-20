---
library_name: transformers
license: apache-2.0
base_model: bert-base-cased
tags:
- generated_from_trainer
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: bert-finetuned-ner
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-finetuned-ner

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0693
- Precision: 0.9360
- Recall: 0.9469
- F1: 0.9414
- Accuracy: 0.9853

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0807        | 1.0   | 1756 | 0.0716          | 0.9014    | 0.9251 | 0.9131 | 0.9805   |
| 0.0357        | 2.0   | 3512 | 0.0710          | 0.9387    | 0.9436 | 0.9412 | 0.9846   |
| 0.0222        | 3.0   | 5268 | 0.0693          | 0.9360    | 0.9469 | 0.9414 | 0.9853   |


### Framework versions

- Transformers 4.57.0
- Pytorch 2.8.0
- Datasets 4.2.0
- Tokenizers 0.22.1
