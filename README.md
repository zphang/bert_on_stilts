# BERT on STILTs

*STILTs = **S**upplementary **T**raining on **I**ntermediate **L**abeled-data **T**asks*

This repository contains code for [BERT on STILTs](https://arxiv.org/abs/1811.01088v2). It is a fork of the [Hugging Face implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

STILTs is a method for supplementary training on an intermediate task before fine-tuning for a downstream target task. We show in [our paper](https://arxiv.org/abs/1811.01088v2) that can improve the performance and stability of the final model.

**BERT on STILTs** achieves a [GLUE score](https://gluebenchmark.com/leaderboard) of 82.0, compared to 80.5 of BERT without STILTs.

## Trained Models

*Coming: 03/01/2019*

| Base Model | Intermediate Task | Target Task | Download | Val Score |
| :---: | :---: | :---: | :---: | :---: |
| BERT-Large   | N/A      | **CoLA**   | Link | - |
| BERT-Large   | **MNLI** | **SST**    | Link | - |
| BERT-Large   | **MNLI** | **MRPC**   | Link | - |
| BERT-Large   | N/A      | **QQP**    | Link | - |
| BERT-Large   | **MNLI** | **STS-B**  | Link | - |
| BERT-Large   | N/A      | **MNLI**   | Link | - |
| BERT-Large   | **MNLI** | **QNLI**   | Link | - |
| BERT-Large   | **MNLI** | **RTE**    | Link | - |
 
*Models differ slightly from published results because they were retrained.*

## Example usage

#### Preparation

You will need to download the GLUE data to run our tasks. See [here](https://github.com/jsalt18-sentence-repl/jiant#downloading-data).

You will also need to set the two following environment variables:

* `GLUE_DIR`: This should point to the location of the GLUE data downloaded from `jiant`.
* `BERT_ALL_DIR`: This should point to the location of BERT downloaded from [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip). Importantly, the `BERT_ALL_DIR` needs to contain the files `uncased_L-24_H-1024_A-16/bert_config.json` and `uncased_L-24_H-1024_A-16/vocab.txt`.

##### Example 1: Generating Predictions

To generate validation/test predictions, as well as validation metrics, run something like the following:

```bash
export TASK=rte
export BERT_LOAD_PATH=path/to/mnli__rte.p
export OUTPUT_PATH=rte_output

python glue/train.py \
    --task_name $TASK \
    --do_val --do_test \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_mode model_only \
    --bert_load_path $BERT_LOAD_PATH \
    --train_batch_size 8 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH
``` 

##### Example 2: Fine-tuning from vanilla BERT

We recommend training with a batch size of 16/24/32.

```bash
export TASK=mnli
export OUTPUT_PATH=mnli

python glue/train.py \
    --task_name $TASK \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_mode from_pretrained \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH
``` 


##### Example 3: STILTs MNLI &rarr; RTE 

```bash
export TASK_A=mnli
export TASK_B=rte
export OUTPUT_PATH_A=mnli
export OUTPUT_PATH_B=mnli__rte

# MNLI
python glue/train.py \
    --task_name $TASK_A \
    --do_train --do_val \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_mode from_pretrained \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH_A
    
# MNLI -> RTE
python glue/train.py \
    --task_name $TASK_B \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_path $OUTPUT_PATH_A/all_state.p
    --bert_load_mode model_only \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH_B
``` 


## FAQ

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

## Citation

```
@article{phang2018stilts,
  title={Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks},
  author={Phang, Jason and Févry,, Thibault and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:1811.01088v2},
  year={2018}
}
```