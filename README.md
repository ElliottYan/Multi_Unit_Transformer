# Multi-Unit Transformers for Neural Machine Translation

This repo is the source code for EMNLP 2020 main conference paper:

[Multi-Unit Transformers for Neural Machine Translation](https://arxiv.org/abs/2010.10743)

The code base is based on OpenNMT-py [https://github.com/OpenNMT/OpenNMT-py]. We will update scripts in the next few days.

## Dependencies

For dependencies, we refer readers to https://github.com/OpenNMT/OpenNMT-py. 


## Datasets
In our paper, we use NIST Chinese-English, WMT'14 English-German and WMT'18 Chinese-English, which are widely used in machine translation studies.

NIST dataset is not publically available.  
For WMT'14 and WMT'18, we refer readers to http://www.statmt.org/wmt14/ and http://www.statmt.org/wmt19/ for downloading. 
We clean, tokenized and apply bpe to these datasets. The clean scripts are similar to script here [https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md].