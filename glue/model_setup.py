import os
import torch

from pytorch_pretrained_bert.modeling import (
    BertConfig, BertForSequenceClassification, BertForSequenceRegression,
    load_from_adapter,
)
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.tokenization import (
    BertTokenizer, PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP,
)
import pytorch_pretrained_bert.utils as utils

from glue.tasks import TaskType


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


def create_model(task_type, bert_model_name, bert_load_mode, bert_load_args,
                 all_state,
                 num_labels, device, n_gpu, fp16, local_rank,
                 bert_config_json_path=None):
    if bert_load_mode == "from_pretrained":
        assert all_state is None
        assert bert_config_json_path is None
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank)
        model = create_from_pretrained(
            task_type=task_type,
            bert_model_name=bert_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model"]:
        model = load_bert(
            task_type=task_type,
            bert_model_name=bert_model_name,
            bert_load_mode=bert_load_mode,
            all_state=all_state,
            num_labels=num_labels,
            bert_config_json_path=bert_config_json_path,
        )
    elif bert_load_mode in ["state_adapter"]:
        model = load_bert_adapter(
            task_type=task_type,
            bert_model_name=bert_model_name,
            bert_load_mode=bert_load_mode,
            bert_load_args=bert_load_args,
            all_state=all_state,
            num_labels=num_labels,
            bert_config_json_path=bert_config_json_path,
        )
    else:
        raise KeyError(bert_load_mode)
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


def create_from_pretrained(task_type, bert_model_name, cache_dir, num_labels):
    if task_type == TaskType.CLASSIFICATION:
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=bert_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )
    elif task_type == TaskType.REGRESSION:
        assert num_labels == 1
        model = BertForSequenceRegression.from_pretrained(
            pretrained_model_name_or_path=bert_model_name,
            cache_dir=cache_dir,
        )
    else:
        raise KeyError(task_type)
    return model


def load_bert(task_type, bert_model_name, bert_load_mode, all_state, num_labels,
              bert_config_json_path=None):
    if bert_config_json_path is None:
        bert_config_json_path = os.path.join(get_bert_config_path(bert_model_name), "bert_config.json")
    if bert_load_mode == "model_only":
        state_dict = all_state
    elif bert_load_mode in ["state_model_only", "state_all", "state_full_model"]:
        state_dict = all_state["model"]
    else:
        raise KeyError(bert_load_mode)

    if task_type == TaskType.CLASSIFICATION:
        if bert_load_mode == "state_full_model":
            model = BertForSequenceClassification.from_state_dict_full(
                config_file=bert_config_json_path,
                state_dict=state_dict,
                num_labels=num_labels,
            )
        else:
            model = BertForSequenceClassification.from_state_dict(
                config_file=bert_config_json_path,
                state_dict=state_dict,
                num_labels=num_labels,
            )
    elif task_type == TaskType.REGRESSION:
        assert num_labels == 1
        if bert_load_mode == "state_full_model":
            model = BertForSequenceRegression.from_state_dict_full(
                config_file=bert_config_json_path,
                state_dict=state_dict,
            )
        else:
            model = BertForSequenceRegression.from_state_dict(
                config_file=bert_config_json_path,
                state_dict=state_dict,
            )
    else:
        raise KeyError(task_type)
    return model


def load_bert_adapter(task_type, bert_model_name, bert_load_mode, bert_load_args,
                      all_state, num_labels, bert_config_json_path):
    if bert_config_json_path is None:
        bert_config_json_path = os.path.join(get_bert_config_path(bert_model_name), "bert_config.json")

    if bert_load_mode in ["model_only_adapter"]:
        adapter_state = all_state
    elif bert_load_mode in ["state_adapter"]:
        adapter_state = all_state["model"]
    else:
        raise KeyError(bert_load_mode)

    # Format: "bert_model_path:{path}"
    #  Very hackish
    bert_state = torch.load(bert_load_args.replace("bert_model_path:", ""))

    config = BertConfig.from_json_file(bert_config_json_path)
    if task_type == TaskType.CLASSIFICATION:
        model = BertForSequenceClassification(config, num_labels=num_labels)
    elif task_type == TaskType.REGRESSION:
        assert num_labels == 1
        model = BertForSequenceRegression(config)
    else:
        raise KeyError(task_type)

    load_from_adapter(
        model=model,
        bert_state=bert_state,
        adapter_state=adapter_state,
    )

    return model


def create_tokenizer(bert_model_name, bert_load_mode, do_lower_case, bert_vocab_path=None):
    if bert_load_mode == "from_pretrained":
        assert bert_vocab_path is None
        tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
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


def save_bert(model, optimizer, args, save_path, save_mode="all"):
    assert save_mode in [
        "all", "tunable", "model_all", "model_tunable",
    ]

    save_dict = dict()

    # Save args
    save_dict["args"] = vars(args)

    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
    if save_mode in ["all", "model_all"]:
        model_state_dict = model_to_save.state_dict()
    elif save_mode in ["tunable", "model_tunable"]:
        model_state_dict = get_tunable_state_dict(model_to_save)
    else:
        raise KeyError(save_mode)
    print("Saving {} model elems:".format(len(model_state_dict)))
    save_dict["model"] = utils.to_cpu(model_state_dict)

    # Save optimizer
    if save_mode in ["all", "tunable"]:
        optimizer_state_dict = utils.to_cpu(optimizer.state_dict()) if optimizer is not None else None
        print("Saving {} optimizer elems:".format(len(optimizer_state_dict)))

    torch.save(save_dict, save_path)


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
