import os

import torch

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import (
    BertTokenizer, PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP,
)


TF_PYTORCH_BERT_NAME_MAP = {
    "bert-base-uncased": "uncased_L-12_H-768_A-12",
    "bert-large-uncased": "uncased_L-24_H-1024_A-16",
}


def get_bert_config_path(bert_model_name):
    return os.path.join(os.environ["BERT_ALL_DIR"], TF_PYTORCH_BERT_NAME_MAP[bert_model_name])


def load_overall_state(bert_load_path, relaxed=True):
    if bert_load_path is None:
        if relaxed:
            return None
        else:
            raise RuntimeError("Need 'bert_load_path'")
    else:
        return torch.load(bert_load_path)


def create_tokenizer(bert_model_name, bert_load_mode, do_lower_case, bert_vocab_path=None):
    if bert_load_mode == "from_pretrained":
        assert bert_vocab_path is None
        tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                            "full_model_only",
                            "state_adapter"]:
        tokenizer = load_tokenizer(
            bert_model_name=bert_model_name,
            do_lower_case=do_lower_case,
            bert_vocab_path=bert_vocab_path,
        )
    else:
        raise KeyError(bert_load_mode)
    return tokenizer


def load_tokenizer(bert_model_name, do_lower_case, bert_vocab_path=None):
    if bert_vocab_path is None:
        bert_vocab_path = os.path.join(get_bert_config_path(bert_model_name), "vocab.txt")
    max_len = min(PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[bert_model_name], int(1e12))
    tokenizer = BertTokenizer(
        vocab_file=bert_vocab_path,
        do_lower_case=do_lower_case,
        max_len=max_len,
    )
    return tokenizer


def get_opt_train_steps(num_train_examples, args):
    num_train_steps = int(
        num_train_examples
        / args.train_batch_size
        / args.gradient_accumulation_steps
        * args.num_train_epochs,
    )
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    return t_total


def create_optimizer(model, learning_rate, t_total, loss_scale, fp16, warmup_proportion, state_dict):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = [
        'bias', 'LayerNorm.bias', 'LayerNorm.weight',
        'adapter.down_project.weight', 'adapter.up_project.weight',
    ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total)

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
    return optimizer


def stage_model(model, fp16, device, local_rank, n_gpu):
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def get_tunable_state_dict(model, verbose=True):
    # Drop non-trainable params
    # Sort of a hack, because it's not really clear when we want/don't want state params,
    #   But for now, layer norm works in our favor. But this will be annoying.
    model_state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if verbose:
                print("    Skip {}".format(name))
            del model_state_dict[name]
    return model_state_dict
