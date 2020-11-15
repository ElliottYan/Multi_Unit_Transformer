# Multi-Unit Transformers for Neural Machine Translation

This repo is the source code for EMNLP 2020 main conference paper:

[Multi-Unit Transformers for Neural Machine Translation](https://arxiv.org/abs/2010.10743)

The code base is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). We will update scripts in the next few days.

## Dependencies

For dependencies, we refer readers to https://github.com/OpenNMT/OpenNMT-py. 


## Datasets
In our paper, we use NIST Chinese-English, WMT'14 English-German and WMT'18 Chinese-English, which are widely used in machine translation studies.

NIST dataset is not publically available.  
For WMT'14 and WMT'18, we refer readers to http://www.statmt.org/wmt14/ and http://www.statmt.org/wmt19/ for downloading. 
We clean, tokenized and apply bpe to these datasets. The clean scripts are similar to script [here](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md).


## Scripts
To reproduce our results, we recommend you to use OpenNMT original directory to preprocess data. 


### Data preprocess
Should be the same as in OpenNMT. 
Note that we add some customized data preprocessing in our code to adapt new input, which might be a little different from originial OpenNMT file format and are not used in the experiments. It should be working well and we will clean up the codes in the next release version.

### Train Baseline
Note that for different dataset, different configurations are used. The main difference lies in dropout, training step and number of GPU used. 
Here we provide our WMT'14 En-De scripts as examples.

The bash script and config file for train a baseline Transformer model with relative positions are shown below:
```python
    python $opennmt_dir/train.py -config $config_dir/en2de_maxlen_128_rel.config
```

where the en2de_maxlen_128_rel.config is :
```yaml
data: $data_dir
save_model: $save_dir
log_file: $log_dir
tensorboard: 'false'
save_checkpoint_steps: 5000
keep_checkpoint: 40
seed: 3435
train_steps: 300000
valid_steps: 5000
warmup_steps: 16000
report_every: 100

decoder_type: transformer
encoder_type: transformer
enc_layers: 6
dec_layers: 6
word_vec_size: 512
rnn_size: 512
transformer_ff: 2048
heads: 8

accum_count: 1
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 1
learning_rate_decay: 1.0
max_grad_norm: 0.0

batch_size: 4096
batch_type: tokens
normalization: tokens
dropout: 0.2
attention_dropout: 0.0
label_smoothing: 0.1

max_generator_batches: 2

param_init: 0.0
param_init_glorot: 'true'
position_encoding: 'true'
share_decoder_embeddings: 'true'

max_relative_positions: 16

world_size: 4
gpu_ranks:
- 0
- 1
- 2
- 3
```

### Train Our model
To train our MUTE models, we provide our config file to reproduce our results.

MUTE:
```yaml
data: $data_dir
save_model: $save_dir
log_file: $log_dir
tensorboard: 'false'
save_checkpoint_steps: 5000
keep_checkpoint: 20
seed: 3435
train_steps: 300000
valid_steps: 5000
warmup_steps: 16000
report_every: 100

decoder_type: transformer
encoder_type: transformer_boost
enc_layers: 6
dec_layers: 6
word_vec_size: 512
rnn_size: 512
transformer_ff: 2048
heads: 8
num_boost: 4
boost_type: 'identity'
boost_dropout_diff: 0.0
boost_main: 'false'
boost_with_ffn: 'true'
boost_str: 'd4'
disable_learnable_weights: 'false'

accum_count: 1
optim: fusedadam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 1
learning_rate_decay: 1.0
max_grad_norm: 0.0
max_relative_positions: 16

batch_size: 4096
batch_type: tokens
normalization: tokens
dropout: 0.2
attention_dropout: 0.0
label_smoothing: 0.1

max_generator_batches: 0

param_init: 0.0
master_port: 13456
param_init_glorot: 'true'
position_encoding: 'true'
model_dtype: 'fp16'
share_decoder_embeddings: 'true'

world_size: 4
gpu_ranks:
- 0
- 1
- 2
- 3
```

Biased MUTE:
```yaml
data: $data_dir
save_model: $save_dir
log_file: $log_dir
tensorboard: 'false'
save_checkpoint_steps: 5000
keep_checkpoint: 20
seed: 3435
train_steps: 300000
valid_steps: 5000
warmup_steps: 16000
report_every: 100

decoder_type: transformer
encoder_type: transformer_boost
enc_layers: 6
dec_layers: 6
word_vec_size: 512
rnn_size: 512
transformer_ff: 2048
heads: 16
num_boost: 4
boost_type: 'identity'
boost_dropout_diff: 0.00
boost_main: 'false'
boost_with_ffn: 'true'
boost_str: 'd1 a3'
shuffle_merge: 'false'
shuffle_merge_type: 'avg'
disable_learnable_weights: 'false'
extra_loss_weight: 1.0
boost_adv_method_list: ["swap","mask","reorder"]
enable_adversarial_training: 'false'
adv_gradient_boost: 'false'
adv_gradient_boost_step: 1.0
adv_gradient_boost_func: cos
boost_sample_rate: 0.85

max_relative_positions: 16

accum_count: 1
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 1
learning_rate_decay: 1.0
max_grad_norm: 0.0

batch_size: 4096
batch_type: tokens
normalization: tokens
dropout: 0.2
attention_dropout: 0.0
label_smoothing: 0.1

max_generator_batches: 0

param_init: 0.0
master_port: 25699
param_init_glorot: 'true'
position_encoding: 'true'
share_decoder_embeddings: 'true'

world_size: 4
gpu_ranks:
- 0
- 1
- 2
- 3
```

Sequentially Biased MUTE:
```yaml
data: $data_dir
save_model: $save_dir
log_file: $log_dir
tensorboard: 'false'
save_checkpoint_steps: 5000
keep_checkpoint: 20
seed: 3435
train_steps: 300000
valid_steps: 5000
warmup_steps: 16000
report_every: 100

decoder_type: transformer
encoder_type: transformer_boost
enc_layers: 6
dec_layers: 6
word_vec_size: 512
rnn_size: 512
transformer_ff: 2048
heads: 16
num_boost: 4
boost_type: 'identity'
boost_dropout_diff: 0.00
boost_main: 'false'
boost_with_ffn: 'true'
boost_str: 'd1 a3'
shuffle_merge: 'true'
shuffle_merge_type: 'avg'
disable_learnable_weights: 'false'
extra_loss_weight: 1.0
boost_adv_method_list: ["swap","mask","reorder"]
enable_adversarial_training: 'false'
adv_gradient_boost: 'false'
adv_gradient_boost_step: 1.0
adv_gradient_boost_func: cos
boost_sample_rate: 0.85

max_relative_positions: 16

accum_count: 1
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 1
learning_rate_decay: 1.0
max_grad_norm: 0.0

batch_size: 4096
batch_type: tokens
normalization: tokens
dropout: 0.2
attention_dropout: 0.0
label_smoothing: 0.1

max_generator_batches: 0

param_init: 0.0
master_port: 25699
param_init_glorot: 'true'
position_encoding: 'true'
share_decoder_embeddings: 'true'

world_size: 4
gpu_ranks:
- 0
- 1
- 2
- 3
```