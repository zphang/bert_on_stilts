import random
import logging

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from .runners import (
    GlueTaskRunner, InputExample, InputFeatures, HybridLoader,
    convert_to_dataset, tokenize_example, _truncate_seq_pair,
)

logger = logging.getLogger(__name__)


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

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
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_feature(example, tokenizer, max_seq_length, label_map, do_tokenize=True,
                               mask_a=False, mask_b=False):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if mask_a:
        tokens_a, _ = random_word(tokens_a, tokenizer)
    if mask_b and tokens_b:
        tokens_b, _ = random_word(tokens_a, tokenizer)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    return InputFeatures(
        guid=example,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        tokens=tokens,
    )


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, verbose=True,
                                 mask_a=False, mask_b=False):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
            mask_a=mask_a, mask_b=mask_b,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))

        features.append(feature_instance)
    return features


class MaskedGlueTaskRunner(GlueTaskRunner):

    def __init__(self, *args, **kwargs):
        self.mask_a = kwargs.get("mask_a", True)
        self.mask_b = kwargs.get("mask_b", True)
        if "mask_a" in kwargs:
            del kwargs["mask_a"]
        if "mask_b" in kwargs:
            del kwargs["mask_b"]
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self, train_examples):
        train_features = convert_examples_to_features(
            train_examples, self.label_map, self.rparams.max_seq_length, self.tokenizer,
            mask_a=self.mask_a, mask_b=self.mask_b,
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
