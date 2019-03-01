import collections as col
import logging
import numpy as np
import random
from tqdm import tqdm, trange

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .core import InputFeatures, Batch, InputExample, TokenizedExample, random_word
from pytorch_pretrained_bert.utils import truncate_seq_pair
from shared.runners import warmup_linear

logger = logging.getLogger(__name__)


def tokenize_example(example, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    else:
        tokens_b = example.text_b
    return TokenizedExample(
        guid=example.guid,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        is_next=example.is_next,
    )


def convert_example_to_features(example, tokenizer, max_seq_length, select_prob=0.15):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer, select_prob=select_prob)
    tokens_b, t2_label = random_word(tokens_b, tokenizer, select_prob=select_prob)
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

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

    features = InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        is_next=example.is_next,
        tokens=tokens,
    )
    return features


def convert_examples_to_features(examples, max_seq_length, tokenizer,
                                 select_prob=0.15, verbose=True):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        feature_instance = convert_example_to_features(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            select_prob=select_prob,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("is_next: %s" % example.is_next)
            logger.info("lm_label_ids: %s " % " ".join([
                str(x) for x in feature_instance.lm_label_ids]))

        features.append(feature_instance)
    return features


def convert_to_dataset(features):
    full_batch = features_to_data(features)
    dataset_ls = [full_batch.input_ids, full_batch.input_mask,
                  full_batch.segment_ids, full_batch.lm_label_ids]
    if full_batch.is_next is not None:
        dataset_ls.append(full_batch.is_next)

    dataset = TensorDataset(*dataset_ls)
    return dataset, full_batch.tokens


def features_to_data(features):
    if features[0].is_next is not None:
        is_next = torch.tensor([f.is_next for f in features], dtype=torch.long)
    else:
        is_next = None
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        lm_label_ids=torch.tensor([f.lm_label_ids for f in features], dtype=torch.long),
        is_next=is_next,
        tokens=[f.tokens for f in features],
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 4:
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            elif len(batch) == 3:
                input_ids, input_mask, segment_ids, lm_label_ids = batch
                is_next = None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i+1) * batch_size]
            yield Batch(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_label_ids=lm_label_ids,
                is_next=is_next,
                tokens=batch_tokens,
            )

    def __len__(self):
        return len(self.dataloader)


"""
class LMDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8",
                 corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line
"""


class LMDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = []
                        #remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        #store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        t1, t2, is_next_label = self.random_sent(item)

        """
        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = self.tokenizer.tokenize(t2)

        # combine to one sample
        cur_example = TokenizedExample(
            guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label,
        )

        # transform sample to features
        cur_features = convert_example_to_features(
            example=cur_example,
            max_seq_length=self.seq_len,
            tokenizer=self.tokenizer,
        )

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next))
        return cur_tensors
        """
        return InputExample(
            guid=cur_id,
            text_a=t1,
            text_b=t2,
            is_next=is_next_label,
        )

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = next(self.file).strip()
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t2

        assert t1 != ""
        assert t2 != ""
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class TrainEpochState:
    def __init__(self):
        self.tr_loss = 0
        self.global_step = 0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0


class RunnerParameters:
    def __init__(self, select_prob, max_seq_length, local_rank, n_gpu, fp16,
                 learning_rate, gradient_accumulation_steps, t_total, warmup_proportion,
                 num_train_epochs, train_batch_size):
        self.select_prob = select_prob
        self.max_seq_length = max_seq_length
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size


class LMRunner:
    def __init__(self, model, optimizer, tokenizer, device, rparams):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.rparams = rparams

    def run_train(self, train_examples, verbose=True):
        if verbose:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader = self.get_train_dataloader(train_examples, verbose=verbose)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)

    def run_train_val(self, train_examples, val_examples):
        epoch_result_dict = col.OrderedDict()
        for i in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            train_dataloader = self.get_train_dataloader(train_examples, verbose=False)
            self.run_train_epoch(train_dataloader)
            epoch_result = self.run_val(val_examples, verbose=False)
            del epoch_result["logits"]
            epoch_result_dict[i] = epoch_result
        return epoch_result_dict

    def run_train_epoch(self, train_dataloader):
        for _ in self.run_train_epoch_context(train_dataloader):
            pass

    def run_train_epoch_context(self, train_dataloader):
        self.model.train()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state):
        batch = batch.to(self.device)
        loss = self.model(
            batch.input_ids, batch.segment_ids, batch.input_mask,
            batch.lm_label_ids, batch.is_next,
        )
        if self.rparams.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        if self.rparams.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += batch.input_ids.size(0)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            if self.rparams.fp16:
                lr_this_step = self.rparams.learning_rate * warmup_linear(
                    train_epoch_state.global_step / self.rparams.t_total, self.rparams.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1

    def run_val(self, val_examples, verbose=True):
        val_dataloader = self.get_eval_dataloader(val_examples, verbose=verbose)
        self.model.eval()
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            with torch.no_grad():
                tmp_eval_loss = self.model(
                    batch.input_ids, batch.segment_ids, batch.input_mask,
                    batch.lm_label_ids, batch.is_next,
                )
                logits = self.model(batch.input_ids, batch.segment_ids, batch.input_mask)

            logits = logits.detach().cpu().numpy()
            total_eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch.input_ids.size(0)
            nb_eval_steps += 1
            all_logits.append(logits)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
        }

    def run_test(self, test_examples, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_examples, verbose=verbose)
        self.model.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Predictions (Test)")):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.model(batch.input_ids, batch.segment_ids, batch.input_mask)
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples, verbose=True):
        train_features = convert_examples_to_features(
            examples=train_examples,
            max_seq_length=self.rparams.max_seq_length,
            tokenizer=self.tokenizer,
            select_prob=self.rparams.select_prob,
            verbose=verbose,
        )
        train_data, train_tokens = convert_to_dataset(train_features)
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
            examples=eval_examples,
            max_seq_length=self.rparams.max_seq_length,
            tokenizer=self.tokenizer,
            select_prob=self.rparams.select_prob,
            verbose=verbose,
        )
        eval_data, eval_tokens = convert_to_dataset(eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(eval_dataloader, eval_tokens)
