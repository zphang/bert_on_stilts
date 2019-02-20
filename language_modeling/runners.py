import logging

import torch
from torch.utils.data import TensorDataset

from .core import InputFeatures, Batch, InputExample, TokenizedExample, random_word
from pytorch_pretrained_bert.utils import truncate_seq_pair

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
        lm_labels=example.lm_labels,
    )


def convert_example_to_feature(example, tokenizer, max_seq_length, select_prob=0.15):
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
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature(
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
            logger.info("is_next: %s (id = %d)" % example.is_next)
            logger.info("lm_label_ids: %s " % " ".join([
                str(x) for x in feature_instance.lm_label_ids]))

        features.append(feature_instance)
    return features


def convert_to_dataset(features):
    full_batch = features_to_data(features)
    dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                            full_batch.segment_ids, full_batch.is_next,
                            full_batch.lm_label_ids)
    return dataset, full_batch.tokens


def features_to_data(features):
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        is_next=torch.tensor([f.is_next for f in features], dtype=torch.long),
        lm_label_ids=torch.tensor([f.lm_label_ids for f in features], dtype=torch.long),
        tokens=[f.tokens for f in features],
    )
