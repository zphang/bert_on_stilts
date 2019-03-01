# BERT on STILTs

*STILTs = **S**upplementary **T**raining on **I**ntermediate **L**abeled-data **T**asks*

This repository contains code for [BERT on STILTs](https://arxiv.org/abs/1811.01088v2). It is a fork of the [Hugging Face implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

STILTs is a method for supplementary training on an intermediate task before fine-tuning for a downstream target task. We show in [our paper](https://arxiv.org/abs/1811.01088v2) that can improve the performance and stability of the final model.

**BERT on STILTs** achieves a [GLUE score](https://gluebenchmark.com/leaderboard) of 82.0, compared to 80.5 of BERT without STILTs.

### Trained Models

*Coming: 03/01/2019*

| Base Model | Intermediate Task | Target Task | Download | Val Score |
| :---: | :---: | :---: | :---: | :---: |
| BERT   | N/A      | **CoLA**   | Link | - |
| BERT   | **MNLI** | **SST**    | Link | - |
| BERT   | **MNLI** | **MRPC**   | Link | - |
| BERT   | N/A      | **QQP**    | Link | - |
| BERT   | **MNLI** | **STS-B**  | Link | - |
| BERT   | N/A      | **MNLI**   | Link | - |
| BERT   | **MNLI** | **QNLI**   | Link | - |
| BERT   | **MNLI** | **RTE**    | Link | - |
 
*Models differ slightly from published results because they were retrained.*

### Example usage

*Coming: 03/01/2019*

### FAQ

> What does STILTs stand for?

STILTs stand for ***S**upplementary **T**raining on **I**ntermediate **L**abeled-data **T**asks*.

> That's it? You finetune on one task, then finetune on another?

Yes—in some sense, this is the simplest possible approach to pretrain/multi-task-train on another task. We do not perform multi-task training or balancing of multi-task losses, any additional hyperparameter search, early stopping or any other modification to the training procedure. We simply apply the standard BERT training procedure *twice*.

Even so, this simple method of supplementary training is still competitive with more complex multi-task methods such as MT-DNN and ALICE.

So far, we have observed three main benefits of STILTs:

1. STILTs tends to improve performance on tasks with little data (<10k training examples)
2. STILTs tends to stabilize training on tasks with little data
3. In cases where the intermediate and target tasks are closely related (e.g. MNLI/RTE), we observe a significant improvement in performance. 

> The paper/abstract mentions a GLUE score of 81.8, while the leaderboard shows 82.0. What's going on?

The GLUE benchmark underwent an update wherein QNLI(v1) was replaced by a QNLIv2, which has a different train/val/test split. The GLUE leaderboard currently reports QNLIv2 scores. On the other hand, all experiments in the paper were run on QNLIv1, so we chose to report the GLUE score based on QNLIv1 in the paper.

In short, with QNLIv1, we get a GLUE score of 81.8. With QNLIv2, we got a score of 82.0.

> When I run finetuning on CoLA/MRPC/STS-B/RTE, I get terrible results. Why?

Finetuning BERT-Large on tasks with little training data (<10k) tends to be unstable. This is referenced in the original BERT paper.

> Where are the other models (GPT, ELMo) referenced in the paper?

Those results were obtained using the [jiant](https://github.com/jsalt18-sentence-repl/jiant) framework. We currently have no plans to publish the trained models for those experiments.  

### Citation

```
@article{phang2018stilts,
  title={Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks},
  author={Phang, Jason and Févry,, Thibault and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:1811.01088v2},
  year={2018}
}
```