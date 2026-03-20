---
library_name: transformers
license: mit
base_model: gpt2
tags:
- generated_from_trainer
model-index:
- name: codeparrot-ds
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# codeparrot-ds

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0530

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 256
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 1000
- num_epochs: 1

### Training results

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 2.5489        | 0.0766 | 5000  | 1.7313          |
| 1.6705        | 0.1533 | 10000 | 1.5152          |
| 1.5225        | 0.2299 | 15000 | 1.4116          |
| 1.4435        | 0.3065 | 20000 | 1.3450          |
| 1.3836        | 0.3832 | 25000 | 1.2935          |
| 1.3326        | 0.4598 | 30000 | 1.2465          |
| 1.287         | 0.5365 | 35000 | 1.2038          |
| 1.2409        | 0.6131 | 40000 | 1.1617          |
| 1.2003        | 0.6897 | 45000 | 1.1237          |
| 1.163         | 0.7664 | 50000 | 1.0907          |
| 1.1362        | 0.8430 | 55000 | 1.0678          |
| 1.115         | 0.9196 | 60000 | 1.0557          |
| 1.1077        | 0.9963 | 65000 | 1.0530          |


### Framework versions

- Transformers 4.57.0
- Pytorch 2.8.0
- Datasets 4.2.0
- Tokenizers 0.22.1
