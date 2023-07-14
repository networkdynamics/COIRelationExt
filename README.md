# QA4COI (Question Answering for Conflict of Interest Extraction)

This repository contains the code and dataset for the paper [Who Controlled the Evidence? QA for Disclosure Information Extraction](https://proceedings.mlr.press/v209/hardy23a/hardy23a.pdf).

## Getting Started

To begin with, install dependencies with `pip install -r requirements.txt`.

## QA for Entity Recognition

### Preprocessing

To preprocess the dataset, edit the config file `config\preprocess\default_config.yaml` as the following:
    
```yaml
dataset:
  train_path: '[path to train data]\train.study.h.json'
  valid_path: '[path to valid/test data]\[valid/test].study.h.json'
  train_output_path: '[path to train data]\train.entity.json'
  valid_output_path: '[path to valid/test data]\[valid/test].entity.json'
generate_entities: false
``` 

Then run `python preprocess.py preprocess=default_config.yaml`. This will create the `[train/valid/test].entity.json` files in the specified paths. We have provided the preprocessed files in the `data\public` folder.

### Training

To train the model, edit the config file `config\train\default_config.yaml` as the following:

```yaml
dataset:
  train_path: '[path to train data]/train.entity.json'
  valid_path: '[path to valid/test data]/[valid/test].entity.json'
  weighted_data: False
  weight_alpha: 0.1
  batch_size: 2
  max_token: 512
  num_workers: 8
  two_classes: False
  debug: False
t5:
  model: 'google/flan-t5-base'
max_token: 512
random_seed: 200
gpus: 1
optimizer:
  lr_rate: 3e-4
  eps: 1e-8
min_steps: 10000
min_epochs: 2
max_epochs: 3
accumulate_grad_batches: 16
fp_16: False
max_grad_norm: 0.5
ckpt_prefix: 'std_entity_200_flan_t5_base'
```

Then run `python train.py train=default_config.yaml`. This will train the model and save the checkpoints in the `checkpoints` folder. We have provided the checkpoints in the [google drive](https://drive.google.com/drive/folders/1EWTurbAihxNm2Jldj5ozWLvfEcZTSF6T?usp=sharing) folder.

### Inference

To run inference on the model, edit the config file `config\inference\default_config.yaml` as the following:

```yaml
dataset:
  predict_path: '[path to valid/test data]/[valid/test].entity.json'
  prediction_path: '[path for inference output]/std_entity_200_flan_t5_base.txt'
  batch_size: 12
  max_token: 512
  num_workers: 8
  debug: False
t5:
  model: 'google/flan-t5-base'
gpus: 1
ckpt_path: '[path to checkpoint]/checkpoints/std_entity_200_flan_t5_base.ckpt'
max_length: 512
beam: 5
```

Then run `python inference.py inference=default_config.yaml`. This will create the `entity.txt` file in the specified path. 

## QA for Relation Extraction

### Preprocessing

To preprocess the dataset, edit the config file `config\preprocess\default_config.yaml` as the following:
    
```yaml
dataset:
  train_path: '[path to train data]/train.study.h.json'
  valid_path: '[path to valid/test data]\[valid/test].study.h.json'
  train_output_path: '[path to train data]train.json'
  valid_output_path: '[path to valid/test data]\[valid/test].json'
  prediction_path: ''
  combine: false
  unlabeled: false
generate_entities: false
``` 

Then run `python preprocess.py preprocess=default_config.yaml`. This will create the `[train/valid/test].json` files in the specified paths. 


### Training

To train the model, edit the config file `config\train\default_config.yaml` as the following:

```yaml
dataset:
  train_path: '[path to train data]/train.json'
  valid_path: '[path to valid/test data]/[valid/test].json'
  weighted_data: False
  weight_alpha: 0.0
  batch_size: 4
  max_token: 512
  num_workers: 16
  two_classes: True
  debug: False
t5:
  model: 'google/flan-t5-base'
max_token: 512
random_seed: 200
gpus: 1
optimizer:
  lr_rate: 3e-4
  eps: 1e-8
min_steps: 10000
min_epochs: 1
max_epochs: 1
accumulate_grad_batches: 16
fp_16: False
max_grad_norm: 0.5
ckpt_prefix: 'std_re_two_classes_200_flan_t5_base'
```

Then run `python train.py train=default_config.yaml`. This will train the model and save the checkpoints in the `checkpoints` folder. We have provided the checkpoints in the [google drive](https://drive.google.com/drive/folders/1EWTurbAihxNm2Jldj5ozWLvfEcZTSF6T?usp=sharing) folder.

### Inference

To run inference on the model, edit the config file `config\inference\default_config.yaml` as the following:

```yaml
dataset:
  predict_path: '[path to valid/test data]/[valid/test].json'
  prediction_path: '[path for inference output]/std_re_two_classes_200_flan_t5_base.txt'
  batch_size: 16
  max_token: 512
  num_workers: 16
  debug: False
t5:
  model: 'google/flan-t5-base'
gpus: 1
max_length: 20
beam: 1
ckpt_path: '[path to checkpoint]/checkpoints/std_re_two_classes_200_flan_t5_base.ckpt'
```

Then run `python inference.py inference=default_config.yaml`. This will create the `entity.txt` file in the specified path. 