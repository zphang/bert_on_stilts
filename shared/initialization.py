import json
import numpy as np
import os
import random
import torch


def init_cuda_from_args(args, logger):
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

    return device, n_gpu


def init_seed(args, n_gpu, logger):
    seed = get_seed(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Using seed: {}".format(seed))
    args.seed = seed

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_train_batch_size(args):
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)


def init_check_mode(args):
    if not(args.do_train or args.do_val or args.do_test):
        raise ValueError("At least one of `do_train` or `do_val` or `do_test` must be True.")


def init_output_dir(args):
    if not args.force_overwrite \
            and (os.path.exists(args.output_dir) and os.listdir(args.output_dir)):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)


def get_seed(seed):
    if seed == -1:
        return int(np.random.randint(0, 2**32 - 1))
    else:
        return seed


def save_args(args):
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        f.write(json.dumps(vars(args), indent=2))
