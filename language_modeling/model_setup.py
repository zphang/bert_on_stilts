import os
import torch

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForPreTraining
import pytorch_pretrained_bert.utils as utils
from shared.model_setup import stage_model, get_bert_config_path, get_tunable_state_dict


def create_model(bert_model_name, bert_load_mode, bert_load_args,
                 all_state,
                 device, n_gpu, fp16, local_rank,
                 bert_config_json_path=None):
    if bert_load_mode == "from_pretrained":
        assert all_state is None
        assert bert_config_json_path is None
        assert bert_load_args is None
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank)
        model = create_from_pretrained(
            bert_model_name=bert_model_name,
            cache_dir=cache_dir,
        )
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model"]:
        assert bert_load_args is None
        model = load_bert(
            bert_model_name=bert_model_name,
            bert_load_mode=bert_load_mode,
            all_state=all_state,
            bert_config_json_path=bert_config_json_path,
        )
    elif bert_load_mode in ["state_adapter"]:
        raise NotImplementedError("Adapter")
    else:
        raise KeyError(bert_load_mode)
    model = stage_model(model, fp16=fp16, device=device, local_rank=local_rank, n_gpu=n_gpu)
    return model


def create_from_pretrained(bert_model_name, cache_dir):
    model = BertForPreTraining.from_pretrained(
        pretrained_model_name_or_path=bert_model_name,
        cache_dir=cache_dir,
    )
    return model


def load_bert(bert_model_name, bert_load_mode, all_state,
              bert_config_json_path=None):
    if bert_config_json_path is None:
        bert_config_json_path = os.path.join(get_bert_config_path(bert_model_name), "bert_config.json")
    if bert_load_mode == "model_only":
        state_dict = all_state
    elif bert_load_mode in ["state_model_only", "state_all", "state_full_model"]:
        state_dict = all_state["model"]
    else:
        raise KeyError(bert_load_mode)

    if bert_load_mode == "state_full_model":
        model = BertForPreTraining.from_state_dict_full(
            config_file=bert_config_json_path,
            state_dict=state_dict,
        )
    else:
        model = BertForPreTraining.from_state_dict(
            config_file=bert_config_json_path,
            state_dict=state_dict,
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
