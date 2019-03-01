import argparse
import logging
import os

import shared.initialization as initialization

# Todo: share
import language_modeling.model_setup as lm_model_setup
import language_modeling.runners as lm_runners
import shared.model_setup as shared_model_setup
import shared.log_info as log_info
from pytorch_pretrained_bert.utils import at_most_one_of


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, "
                             "bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    # === Model parameters === #
    parser.add_argument("--bert_load_path", default=None, type=str)
    parser.add_argument("--bert_load_mode", default="from_pretrained", type=str,
                        help="from_pretrained, model_only, state_model_only, state_all")
    parser.add_argument("--bert_load_args", default=None, type=str)
    parser.add_argument("--bert_config_json_path", default=None, type=str)
    parser.add_argument("--bert_vocab_path", default=None, type=str)
    parser.add_argument("--bert_save_mode", default="all", type=str)

    # === Other parameters === #
    parser.add_argument("--select_prob", default=0.15, type=float)
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
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
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
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
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. "
                             "Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--print-trainable-params', action="store_true")
    parser.add_argument('--not-verbose', action="store_true")
    parser.add_argument('--force-overwrite', action="store_true")
    args = parser.parse_args(*in_args)
    return args


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    log_info.print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)

    tokenizer = shared_model_setup.create_tokenizer(
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        do_lower_case=args.do_lower_case,
        bert_vocab_path=args.bert_vocab_path,
    )
    all_state = shared_model_setup.load_overall_state(args.bert_load_path, relaxed=True)
    model = lm_model_setup.create_model(
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        bert_load_args=args.bert_load_args,
        all_state=all_state,
        device=device,
        n_gpu=n_gpu,
        fp16=args.fp16,
        local_rank=args.local_rank,
        bert_config_json_path=args.bert_config_json_path,
    )
    if args.print_trainable_params:
        log_info.print_trainable_params(model)

    train_dataset = lm_runners.LMDataset(
        args.train_file, tokenizer, seq_len=args.max_seq_length,
        corpus_lines=None, on_memory=args.on_memory,
    )
    t_total = shared_model_setup.get_opt_train_steps(
        num_train_examples=len(train_dataset),
        args=args,
    )
    optimizer = shared_model_setup.create_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        t_total=t_total,
        loss_scale=args.loss_scale,
        fp16=args.fp16,
        warmup_proportion=args.warmup_proportion,
        state_dict=all_state["optimizer"] if args.bert_load_mode == "state_all" else None,
    )
    runner = lm_runners.LMRunner(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        device=device,
        rparams=lm_runners.RunnerParameters(
            select_prob=args.select_prob, max_seq_length=args.max_seq_length,
            local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
            learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
            t_total=t_total, warmup_proportion=args.warmup_proportion,
            num_train_epochs=args.num_train_epochs,
            train_batch_size=args.train_batch_size,
        )
    )

    runner.run_train(train_dataset)
    lm_model_setup.save_bert(
        model=model, optimizer=optimizer, args=args,
        save_path=os.path.join(args.output_dir, "all_state.p"),
        save_mode=args.bert_save_mode,
    )


if __name__ == "__main__":
    main()
