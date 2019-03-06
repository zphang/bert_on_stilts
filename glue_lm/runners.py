import collections as col
import logging
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from glue.core import InputExample
from glue.runners import (
    LabelModes, is_null_label_map,
    tokenize_example, truncate_seq_pair,
    warmup_linear, compute_task_metrics,
    get_label_mode,
)
from language_modeling.runners import (
    random_word,
)
from glue_lm.core import Batch, InputFeatures


logger = logging.getLogger(__name__)


class TrainEpochState:
    def __init__(self):
        self.tr_loss = 0
        self.tr_glue_loss = 0
        self.tr_lm_loss = 0
        self.global_step = 0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0


def convert_example_to_features(example, tokenizer, max_seq_length, label_map,
                                select_prob=0.15):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens_a, t1_label = random_word(tokens_a, tokenizer, select_prob=select_prob)
    if tokens_b:
        tokens_b, t2_label = random_word(tokens_b, tokenizer, select_prob=select_prob)
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    else:
        lm_label_ids = ([-1] + t1_label + [-1])


    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if is_null_label_map(label_map):
        glue_label_id = example.label
    else:
        glue_label_id = label_map[example.label]

    features = InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        glue_label_id=glue_label_id,
        lm_label_ids=lm_label_ids,
        tokens=tokens,
    )
    return features


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, verbose=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_features(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.glue_label_id))

        features.append(feature_instance)
    return features


def convert_to_dataset(features, label_mode):
    full_batch = features_to_data(features, label_mode=label_mode)
    if full_batch.glue_label_ids is None:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids)
    else:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids,
                                full_batch.glue_label_ids, full_batch.lm_label_ids)
    return dataset, full_batch.tokens


def features_to_data(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        glue_label_ids=torch.tensor([f.glue_label_id for f in features], dtype=label_type),
        lm_label_ids=torch.tensor([f.lm_label_ids for f in features], dtype=torch.long),
        tokens=[f.tokens for f in features],
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 5:
                input_ids, input_mask, segment_ids, glue_label_ids, lm_label_ids = batch
            elif len(batch) == 3:
                input_ids, input_mask, segment_ids = batch
                glue_label_ids, lm_label_ids = None, None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i+1) * batch_size]
            yield Batch(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                glue_label_ids=glue_label_ids,
                lm_label_ids=lm_label_ids,
                tokens=batch_tokens,
            )

    def __len__(self):
        return len(self.dataloader)


class RunnerParameters:
    def __init__(self, max_seq_length, local_rank, n_gpu, fp16,
                 glue_loss_weight, lm_loss_weight,
                 learning_rate, gradient_accumulation_steps, t_total, warmup_proportion,
                 num_train_epochs, train_batch_size, eval_batch_size):
        self.max_seq_length = max_seq_length
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.glue_loss_weight = glue_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


class GlueLMTaskRunner:
    def __init__(self, glue_lm_model, optimizer, tokenizer, label_list, device, rparams):
        self.glue_lm_model = glue_lm_model
        self.glue_model = glue_lm_model.glue_model
        self.lm_model = glue_lm_model.lm_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams

    def run_train(self, train_examples, task_name, verbose=True):
        if verbose:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader = self.get_train_dataloader(train_examples, verbose=verbose)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader, task_name)

    def run_train_val(self, train_examples, val_examples, task_name):
        epoch_result_dict = col.OrderedDict()
        for i in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            train_dataloader = self.get_train_dataloader(train_examples, verbose=False)
            self.run_train_epoch(train_dataloader, task_name)
            epoch_result = self.run_val(val_examples, task_name, verbose=False)
            del epoch_result["logits"]
            epoch_result_dict[i] = epoch_result
        return epoch_result_dict

    def run_train_epoch(self, train_dataloader, task_name):
        for _ in self.run_train_epoch_context(train_dataloader, task_name):
            pass

    def run_train_epoch_context(self, train_dataloader, task_name):
        self.glue_lm_model.train()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
                task_name=task_name,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state, task_name):
        batch = batch.to(self.device)
        use_cola = task_name == "cola"
        glue_loss, lm_loss = self.glue_lm_model(
            input_ids=batch.input_ids,
            token_type_ids=batch.segment_ids,
            attention_mask=batch.input_mask,
            glue_labels=batch.glue_label_ids,
            masked_lm_labels=batch.lm_label_ids,
            use_lm=True,
            use_cola=use_cola,
        )
        glue_loss = self.rparams.glue_loss_weight * glue_loss
        lm_loss = self.rparams.lm_loss_weight * lm_loss
        loss = glue_loss + lm_loss

        if self.rparams.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        if self.rparams.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.tr_glue_loss += glue_loss.item()
        train_epoch_state.tr_lm_loss += lm_loss.item()
        # print("[TRAIN] ")
        # print("   GLUE: ", train_epoch_state.tr_glue_loss)
        # print("     LM: ", train_epoch_state.tr_lm_loss)
        # print("  TOTAL: ", train_epoch_state.tr_loss)

        train_epoch_state.nb_tr_examples += batch.input_ids.size(0)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = self.rparams.learning_rate * warmup_linear(
                train_epoch_state.global_step / self.rparams.t_total, self.rparams.warmup_proportion)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1

    def run_val(self, val_examples, task_name, verbose=True):
        self.glue_lm_model.eval()
        val_dataloader = self.get_eval_dataloader(val_examples, verbose=verbose)
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        all_labels = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            with torch.no_grad():
                glue_loss = self.glue_model(
                    input_ids=batch.input_ids,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_mask,
                    labels=batch.glue_label_ids,
                )
                glue_logits = self.glue_model(
                    input_ids=batch.input_ids,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_mask,
                )
                label_ids = batch.glue_label_ids.cpu().numpy()

            glue_logits = glue_logits.detach().cpu().numpy()
            total_eval_loss += glue_loss.mean().item()

            nb_eval_examples += batch.input_ids.size(0)
            nb_eval_steps += 1
            all_logits.append(glue_logits)
            all_labels.append(label_ids)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": compute_task_metrics(task_name, all_logits, all_labels),
        }

    def run_test(self, test_examples, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_examples, verbose=verbose)
        self.glue_lm_model.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Predictions (Test)")):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.glue_model(
                    input_ids=batch.input_ids,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_mask,
                )
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples, verbose=True):
        train_features = convert_examples_to_features(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        train_data, train_tokens = convert_to_dataset(
            train_features, label_mode=get_label_mode(self.label_map),
        )
        if self.rparams.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.rparams.train_batch_size,
        )
        return HybridLoader(train_dataloader, train_tokens)

    def get_eval_dataloader(self, eval_examples, verbose=True):
        eval_features = convert_examples_to_features(
            eval_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            verbose=verbose,
        )
        eval_data, eval_tokens = convert_to_dataset(
            eval_features, label_mode=get_label_mode(self.label_map),
        )
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(eval_dataloader, eval_tokens)
