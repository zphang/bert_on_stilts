import argparse
import torch
import json
import numpy as np
import os
import pandas as pd
import random

import logging

from glue.tasks import PROCESSORS, DEFAULT_FOLDER_NAMES
from glue.runners import GlueTaskRunner, RunnerParameters
from glue import model_setup


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # === Model parameters === #
    parser.add_argument("--bert_load_path", default=None, type=str)
    parser.add_argument("--bert_load_mode", default="from_pretrained", type=str,
                        help="from_pretrained, model_only, state_model_only, state_all")
    parser.add_argument("--bert_config_json_path", default=None, type=str)
    parser.add_argument("--bert_vocab_path", default=None, type=str)

    # === Other parameters === #
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_val",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--print-trainable-params', action="store_true")
    args = parser.parse_args(*in_args)
    return args


def main():
    args = get_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not(args.do_train or args.do_val or args.do_test):
        raise ValueError("At least one of `do_train` or `do_val` or `do_test` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    task_processor = PROCESSORS[task_name]()
    if args.data_dir is None:
        data_dir = os.path.join(os.environ["GLUE_DIR"], DEFAULT_FOLDER_NAMES[task_name])
    else:
        data_dir = args.data_dir

    tokenizer = model_setup.create_tokenizer(
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        do_lower_case=args.do_lower_case,
        bert_vocab_path=args.bert_vocab_path,
    )
    all_state = model_setup.load_overall_state(args.bert_load_path, relaxed=True)
    model = model_setup.create_model(
        task_type=task_processor.TASK_TYPE,
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        all_state=all_state,
        num_labels=len(task_processor.get_labels()),
        device=device,
        n_gpu=n_gpu,
        fp16=args.fp16,
        local_rank=args.local_rank,
        bert_config_json_path=args.bert_config_json_path,
    )
    if args.do_train:
        if args.print_trainable_params:
            print("TRAINABLE PARAMS:")
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    print("    {}".format(param_name))
        train_examples = task_processor.get_train_examples(data_dir)
        num_train_steps = int(
            len(train_examples)
            / args.train_batch_size
            / args.gradient_accumulation_steps
            * args.num_train_epochs,
        )
        t_total = num_train_steps
        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        optimizer = model_setup.create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            t_total=t_total,
            loss_scale=args.loss_scale,
            fp16=args.fp16,
            warmup_proportion=args.warmup_proportion,
            state_dict=all_state["optimizer"] if args.bert_load_mode == "state_all" else None,
        )
    else:
        train_examples = None
        num_train_steps = 0
        t_total = 0
        optimizer = None

    runner = GlueTaskRunner(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        label_list=task_processor.get_labels(),
        device=device,
        rparams=RunnerParameters(
            max_seq_length=args.max_seq_length,
            local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
            learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
            t_total=t_total, warmup_proportion=args.warmup_proportion,
            num_train_epochs=args.num_train_epochs, num_train_steps=num_train_steps,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        )
    )

    if args.do_train:
        runner.run_train(train_examples)

    if args.do_save:
        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "all_state.p")
        torch.save({
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, output_model_file)

    if args.do_val:
        val_examples = task_processor.get_dev_examples(data_dir)
        results = runner.run_val(val_examples, task_name=task_name)
        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "val_preds.csv"), header=False, index=False)
        metrics_str = json.dumps({"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
            f.write(metrics_str)

        # HACK for MNLI-mismatched
        if task_name == "mnli":
            mm_val_examples = PROCESSORS["mnli-mm"].get_dev_examples(data_dir)
            mm_results = runner.run_val(mm_val_examples, task_name=task_name)
            df = pd.DataFrame(results["logits"])
            df.to_csv(os.path.join(args.output_dir, "mm_val_preds.csv"), header=False, index=False)
            combined_metrics = {}
            for k, v in results["metrics"]:
                combined_metrics[k] = v
            for k, v in mm_results["metrics"]:
                combined_metrics["mm-"+k] = v
            combined_metrics_str = json.dumps({
                "loss": results["loss"],
                "metrics": combined_metrics,
            }, indent=2)
            with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
                f.write(combined_metrics_str)

    if args.do_test:
        test_examples = task_processor.get_test_examples(data_dir)
        logits = runner.run_test(test_examples)
        df = pd.DataFrame(logits)
        df.to_csv(os.path.join(args.output_dir, "test_preds.csv"), header=False, index=False)

        # HACK for MNLI-mismatched
        if task_name == "mnli":
            test_examples = PROCESSORS["mnli-mm"].get_test_examples(data_dir)
            logits = runner.run_test(test_examples)
            df = pd.DataFrame(logits)
            df.to_csv(os.path.join(args.output_dir, "mm_test_preds.csv"), header=False, index=False)


if __name__ == "__main__":
    main()
