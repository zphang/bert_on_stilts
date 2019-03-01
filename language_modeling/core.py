import logging
import random

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, text_a, text_b, is_next=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            is_next:
            lm_labels:
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.is_next = is_next  # nextSentence

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "is_next": self.is_next,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class TokenizedExample(object):
    def __init__(self, guid, tokens_a, tokens_b, is_next=None, lm_labels=None):
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "tokens_a": self.tokens_a,
            "tokens_b": self.tokens_b,
            "is_next": self.is_next,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, is_next, lm_label_ids, tokens):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.tokens = tokens

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "is_next": self.is_next,
            "lm_label_ids": self.lm_label_ids,
            "tokens": self.tokens,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


def random_word(tokens, tokenizer, select_prob=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param select_prob: Probability of selecting for prediction
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    tokens = list(tokens)

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < select_prob:
            prob /= select_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


class Batch:
    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.tokens = tokens

    def to(self, device):
        return Batch(
            input_ids=self.input_ids.to(device),
            input_mask=self.input_mask.to(device),
            segment_ids=self.segment_ids.to(device),
            is_next=self.is_next.to(device),
            lm_label_ids=self.lm_label_ids.to(device),
            tokens=self.tokens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, key):
        return Batch(
            input_ids=self.input_ids[key],
            input_mask=self.input_mask[key],
            segment_ids=self.segment_ids[key],
            is_next=self.is_next[key],
            lm_label_ids=self.lm_label_ids[key],
            tokens=self.tokens[key],
        )
