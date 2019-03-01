import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import pytorch_pretrained_bert.utils as utils
from shared.model_setup import get_tunable_state_dict
import pytorch_pretrained_bert.modeling as modeling


"""
def save_bert(glue_model, lm_model, optimizer, args, save_path, save_mode="all", verbose=True):
    assert save_mode in [
        "all", "tunable", "model_all", "model_tunable",
    ]

    save_dict = dict()

    # Save args
    save_dict["args"] = vars(args)

    # Save model
    glue_model_to_save = glue_model.module \
        if hasattr(glue_model, 'module') \
        else glue_model  # Only save the model itself
    if save_mode in ["all", "model_all"]:
        glue_model_state_dict = glue_model_to_save.state_dict()
        lm_state_dict = get_lm_cls_state_dict(lm_model)
    elif save_mode in ["tunable", "model_tunable"]:
        glue_model_state_dict = get_tunable_state_dict(glue_model_to_save)
        lm_state_dict = get_tunable_state_dict(get_lm_cls_state_dict(lm_model))
    else:
        raise KeyError(save_mode)
    if verbose:
        print("Saving {} glue model elems:".format(len(glue_model_state_dict)))
        print("Saving {} lm model elems:".format(len(lm_state_dict)))
    save_dict["model"] = utils.to_cpu(glue_model_state_dict)
    save_dict["lm_model"] = utils.to_cpu(lm_state_dict)

    # Save optimizer
    if save_mode in ["all", "tunable"]:
        optimizer_state_dict = utils.to_cpu(optimizer.state_dict()) if optimizer is not None else None
        if verbose:
            print("Saving {} optimizer elems:".format(len(optimizer_state_dict)))

    torch.save(save_dict, save_path)
"""


def save_bert(glue_lm_model, optimizer, args, save_path, save_mode="all", verbose=True):
    glue_model = glue_lm_model.glue_model
    lm_model = glue_lm_model.lm_model
    assert save_mode in [
        "all", "tunable", "model_all", "model_tunable",
    ]

    save_dict = dict()

    # Save args
    save_dict["args"] = vars(args)

    # Save model
    glue_model_to_save = glue_model.module \
        if hasattr(glue_model, 'module') \
        else glue_model  # Only save the model itself
    if save_mode in ["all", "model_all"]:
        glue_model_state_dict = glue_model_to_save.state_dict()
        lm_state_dict = get_lm_cls_state_dict(lm_model)
    elif save_mode in ["tunable", "model_tunable"]:
        glue_model_state_dict = get_tunable_state_dict(glue_model_to_save)
        lm_state_dict = get_tunable_state_dict(get_lm_cls_state_dict(lm_model))
    else:
        raise KeyError(save_mode)
    if verbose:
        print("Saving {} glue model elems:".format(len(glue_model_state_dict)))
        print("Saving {} lm model elems:".format(len(lm_state_dict)))
    save_dict["model"] = utils.to_cpu(glue_model_state_dict)
    save_dict["lm_model"] = utils.to_cpu(lm_state_dict)

    # Save optimizer
    if save_mode in ["all", "tunable"]:
        optimizer_state_dict = utils.to_cpu(optimizer.state_dict()) if optimizer is not None else None
        if verbose:
            print("Saving {} optimizer elems:".format(len(optimizer_state_dict)))

    torch.save(save_dict, save_path)


def get_lm_cls_state_dict(lm_model):
    lm_state_dict = lm_model.state_dict()
    for k in list(lm_state_dict):
        if k.startswith("bert."):
            del lm_state_dict[k]
    return lm_state_dict


class BertGlueLM(nn.Module):
    def __init__(self, glue_model, lm_model):
        super().__init__()
        self.glue_model = glue_model
        self.lm_model = lm_model

        if isinstance(self.glue_model, modeling.BertForSequenceClassification):
            self._is_classifier = True
            self._glue_loss_fct = CrossEntropyLoss()
        elif isinstance(self.glue_model, modeling.BertForSequenceRegression):
            self._is_classifier = False
            self._glue_loss_fct = MSELoss()
        else:
            raise RuntimeError()

        self.bert = self.glue_model.bert

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                glue_labels=None, masked_lm_labels=None, use_lm=False, use_cola=True):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)

        if self._is_classifier:
            dropout_pooled_output = self.glue_model.dropout(pooled_output)
            glue_output = self.glue_model.classifier(dropout_pooled_output)
        else:
            dropout_pooled_output = self.glue_model.dropout(pooled_output)
            glue_output = self.glue_model.regressor(dropout_pooled_output)

        if glue_labels is not None:
            if self._is_classifier:
                glue_loss = self._glue_loss_fct(glue_output.view(-1, self.glue_model.num_labels),
                                                glue_labels.view(-1))
            else:
                glue_loss = self._glue_loss_fct(glue_output.view(-1), glue_labels.view(-1))

        if not use_lm:
            if glue_labels is not None:
                return glue_loss
            else:
                glue_output
        else:
            prediction_scores, seq_relationship_score = self.lm_model.cls(sequence_output, pooled_output)
            if masked_lm_labels is not None:
                lm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                if use_cola:
                    # CoLA hack
                    selector = glue_labels.byte()
                    masked_lm_loss = lm_loss_fct(
                        prediction_scores[selector].view(-1, self.lm_model.config.vocab_size),
                        masked_lm_labels[selector].view(-1),
                    )
                    masked_lm_loss *= glue_labels.float().mean()
                else:
                    masked_lm_loss = lm_loss_fct(
                        prediction_scores.view(-1, self.lm_model.config.vocab_size), masked_lm_labels.view(-1)
                    )
                assert glue_labels is not None
                return glue_loss, masked_lm_loss
            else:
                assert glue_labels is None
                return glue_output, prediction_scores


