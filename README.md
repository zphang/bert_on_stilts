# BERT on STILTs

This repository contains code for [BERT on STILTs](https://arxiv.org/abs/1811.01088v2). It is is a fork of the [Hugging Face implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

STILTs is a method for supplementary training on an intermediate task before fine-tuning for a downstream target task. We show in [our paper](https://arxiv.org/abs/1811.01088v2) that can improve the performance and stability of the final model.

**BERT on STILTs** achieves a [GLUE score](https://gluebenchmark.com/leaderboard) of 82.0, compared to 80.5 of BERT without STILTs.

### Trained Models

*Coming: 03/01/2019*

* BERT &rarr; **CoLA**: [Link]
* BERT &rarr; MNLI &rarr; **SST**: [Link]
* BERT &rarr; MNLI &rarr; **MRPC**: [Link]
* BERT &rarr; **QQP**: [Link]
* BERT &rarr; MNLI &rarr; **STS-B**: [Link]
* BERT &rarr; **MNLI**: [Link]
* BERT &rarr; MNLI &rarr; **QNLI**: [Link]
* BERT &rarr; MNLI &rarr; **RTE**: [Link]

### Example usage

*Coming: 03/01/2019*

### Citation

```
@article{devlin2018bert,
  title={Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks},
  author={Phang, Jason and Févry,, Thibault and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:1811.01088v2},
  year={2018}
}
```