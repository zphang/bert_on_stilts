# BERT on STILTs

*STILTs = **S**upplementary **T**raining on **I**ntermediate **L**abeled-data **T**asks*

This repository contains code for [BERT on STILTs](https://arxiv.org/abs/1811.01088v2). It is a fork of the [Hugging Face implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

STILTs is a method for supplementary training on an intermediate task before fine-tuning for a downstream target task. We show in [our paper](https://arxiv.org/abs/1811.01088v2) that STILTs can improve the performance and stability of the final model on the target task.

**BERT on STILTs** achieves a [GLUE score](https://gluebenchmark.com/leaderboard) of 82.0, compared to 80.5 of BERT without STILTs.

## Trained Models

| Base Model | Intermediate Task | Target Task | Download | Val Score | Test Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| BERT-Large   | -        | **CoLA**   | [Link](https://drive.google.com/file/d/1bYuvIrnYjI-22xd6koYdDlkgLMtN6Uey/view?usp=sharing) | 65.3 | 61.2 |
| BERT-Large   | **MNLI** | **SST**    | [Link](https://drive.google.com/file/d/1M0ubTzGO4oNC7szc6bRxMIf81iTWgAPL/view?usp=sharing) | 93.9 | 95.1 |
| BERT-Large   | **MNLI** | **MRPC**   | [Link](https://drive.google.com/file/d/1b0FdK-95yLk_P2ro009opSRX6GgwegGB/view?usp=sharing) | 90.4 | 88.6 |
| BERT-Large   | **MNLI** | **STS-B**  | [Link](https://drive.google.com/file/d/1VWZbqFvM2myLoE2-uVh-KtAUmhgS9anb/view?usp=sharing) | 90.7 | 89.0 |
| BERT-Large   | -        | **QQP**    | [Link](https://drive.google.com/file/d/1d5KMckz2txwYtE_wGL6g8591nGFw9Vid/view?usp=sharing) | 90.0 | 81.2 |
| BERT-Large   | -        | **MNLI**   | [Link](https://drive.google.com/file/d/1na4cULKs5qe9odhF0qA-x4H2ZZKXNl7N/view?usp=sharing) | 86.7 | 86.2 | 
| BERT-Large   | **MNLI** | **QNLI**   | [Link](https://drive.google.com/file/d/1cHehR1PXxQ38UrKBdzwUCykZcKoIUeCv/view?usp=sharing) | 92.3 | 92.8 |
| BERT-Large   | **MNLI** | **RTE**    | [Link](https://drive.google.com/file/d/1YIYiqcBTXRCMh8gvKnGCO0mXuhR6PnKF/view?usp=sharing) | 84.1 | 79.0 |
| BERT-Large   | -        | WNLI*     | N/A | 56.3 | 65.1 |

Overall GLUE Score: **82.0**
 
*Models differ slightly from published results because they were retrained.*

## Example usage

#### Preparation

You will need to download the GLUE data to run our tasks. See [here](https://github.com/jsalt18-sentence-repl/jiant#downloading-data).

You will also need to set the two following environment variables:

* `GLUE_DIR`: This should point to the location of the GLUE data downloaded from `jiant`.
* `BERT_ALL_DIR`: Set `BERT_ALL_DIR=/PATH_TO_THIS_REPO/cache/bert_metadata` 
    * For mor general use: `BERT_ALL_DIR` should point to the location of BERT downloaded from [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip). Importantly, the `BERT_ALL_DIR` needs to contain the files `uncased_L-24_H-1024_A-16/bert_config.json` and `uncased_L-24_H-1024_A-16/vocab.txt`.

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
    --bert_load_mode full_model_only \
    --bert_load_path $BERT_LOAD_PATH \
    --eval_batch_size 64 \
    --output_dir $OUTPUT_PATH
``` 

##### Example 2: Fine-tuning from vanilla BERT

We recommend training with a batch size of 16/24/32.

```bash
export TASK=mnli
export OUTPUT_PATH=mnli_output

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


##### Example 3: Fine-tuning from MNLI model

```bash
export PRETRAINED_MODEL_PATH=/path/to/mnli.p
export TASK=rte
export OUTPUT_PATH=rte_output

python glue/train.py \
    --task_name $TASK \
    --do_train --do_val --do_test --do_val_history \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_path $PRETRAINED_MODEL_PATH \
    --bert_load_mode model_only \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH
``` 


##### Example 4: STILTs MNLI &rarr; RTE 

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
    --bert_load_path $OUTPUT_PATH_A/all_state.p \
    --bert_load_mode state_model_only \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH_B
``` 


## Submission to GLUE leaderboard

We have included helper scripts for exporting submissions to the GLUE leaderboard. To prepare for submission, copy the template from `cache/submission_template` to a given new output folder:

```bash
cp -R cache/submission_template /path/to/new_submission
```

After running a fine-tuned/pretrained model on a task with the `--do_test` argument, a folder (e.g. `rte_output`) will be created containing `test_preds.csv` among other files. Run the following command to convert `test_preds.csv` to the submission format in the output folder.

```bash
python glue/format_for_glue.py 
    --task-name rte \
    --input-base-path /path/to/rte_output \
    --output-base-path /path/to/new_submission
```

Once you have exported submission predictions for each task, you should have 11 `.tsv` files in total. If you run `wc -l *.tsv`, you should see something like the following:

```
   1105 AX.tsv
   1064 CoLA.tsv
   9848 MNLI-mm.tsv
   9797 MNLI-m.tsv
   1726 MRPC.tsv
   5464 QNLI.tsv
 390966 QQP.tsv
   3001 RTE.tsv
   1822 SST-2.tsv
   1380 STS-B.tsv
    147 WNLI.tsv
 426597 total 
```

Next run `zip -j -D submission.zip *.tsv` in the folder to generate the submission zip file. Upload the zip file to [https://gluebenchmark.com/submit](https://gluebenchmark.com/submit) to submit to the leaderboard.

## Extras

This repository also supports the use of [Adapter layers](https://arxiv.org/abs/1902.00751) for BERT.

## FAQ

> Q: What does STILTs stand for?

STILTs stand for ***S**upplementary **T**raining on **I**ntermediate **L**abeled-data **T**asks*.

> Q: That's it? You finetune on one task, then finetune on another?

Yes—in some sense, this is the simplest possible approach to pretrain/multi-task-train on another task. We do not perform multi-task training or balancing of multi-task losses, any additional hyperparameter search, early stopping or any other modification to the training procedure. We simply apply the standard BERT training procedure *twice*.

Even so, this simple method of supplementary training is still competitive with more complex multi-task methods such as MT-DNN and ALICE.

So far, we have observed three main benefits of STILTs:

1. STILTs tends to improve performance on tasks with little data (<10k training examples)
2. STILTs tends to stabilize training on tasks with little data
3. In cases where the intermediate and target tasks are closely related (e.g. MNLI/RTE), we observe a significant improvement in performance. 

> Q: The paper/abstract mentions a GLUE score of 81.8, while the leaderboard shows 82.0. What's going on?

The GLUE benchmark underwent an update wherein QNLI(v1) was replaced by a QNLIv2, which has a different train/val/test split. The GLUE leaderboard currently reports QNLIv2 scores. On the other hand, all experiments in the paper were run on QNLIv1, so we chose to report the GLUE score based on QNLIv1 in the paper.

In short, with QNLIv1, we get a GLUE score of 81.8. With QNLIv2, we got a score of 82.0.

> Q: When I run finetuning on CoLA/MRPC/STS-B/RTE, I get terrible results. Why?

Finetuning BERT-Large on tasks with little training data (<10k) tends to be unstable. This is referenced in the original BERT paper.

> Q: Where are the other models (GPT, ELMo) referenced in the paper?

Those results were obtained using the [jiant](https://github.com/jsalt18-sentence-repl/jiant) framework. We currently have no plans to publish the trained models for those experiments.  

## Citation

```
@article{phang2018stilts,
  title={Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks},
  author={Phang, Jason and F\'evry,, Thibault and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:1811.01088v2},
  year={2018}
}
```