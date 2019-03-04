import os
import torch

from pytorch_pretrained_bert.modeling import (
    BertConfig, BertForSequenceClassification, BertForSequenceRegression,
    load_from_adapter,
)
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import pytorch_pretrained_bert.utils as utils
from shared.model_setup import stage_model, get_bert_config_path, get_tunable_state_dict

from glue.tasks import TaskType


def create_model(task_type, bert_model_name, bert_load_mode, bert_load_args,
                 all_state,
                 num_labels, device, n_gpu, fp16, local_rank,
                 bert_config_json_path=None):
    if bert_load_mode == "from_pretrained":
        assert bert_load_args is None
        assert all_state is None
        assert bert_config_json_path is None
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank)
        model = create_from_pretrained(
            task_type=task_type,
            bert_model_name=bert_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                            "full_model_only"]:
        assert bert_load_args is None
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
    model = stage_model(model, fp16=fp16, device=device, local_rank=local_rank, n_gpu=n_gpu)
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
    if bert_load_mode in ("model_only", "full_model_only"):
        state_dict = all_state
    elif bert_load_mode in ["state_model_only", "state_all", "state_full_model"]:
        state_dict = all_state["model"]
    else:
        raise KeyError(bert_load_mode)

    if task_type == TaskType.CLASSIFICATION:
        if bert_load_mode in ("state_full_model", "full_model_only"):
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
        if bert_load_mode in ("state_full_model", "full_model_only"):
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


def save_bert(model, optimizer, args, save_path, save_mode="all", verbose=True):
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
    if verbose:
        print("Saving {} model elems:".format(len(model_state_dict)))
    save_dict["model"] = utils.to_cpu(model_state_dict)

    # Save optimizer
    if save_mode in ["all", "tunable"]:
        optimizer_state_dict = utils.to_cpu(optimizer.state_dict()) if optimizer is not None else None
        if verbose:
            print("Saving {} optimizer elems:".format(len(optimizer_state_dict)))

    torch.save(save_dict, save_path)
