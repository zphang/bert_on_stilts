import torch
from torch.utils.data import Dataset


class IDS:
    UNK = 100
    CLS = 101
    SEP = 102
    MASK = 103


class TensorDatasetPlus(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(
            tensors[0].size(0) == tensor.size(0)
            for tensor in tensors
            if isinstance(tensor, torch.Tensor)
        )
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "label": self.label,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class TokenizedExample(object):
    def __init__(self, guid, tokens_a, tokens_b=None, label=None):
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.label = label

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "tokens_a": self.tokens_a,
            "tokens_b": self.tokens_b,
            "label": self.label,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id, tokens):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.tokens = tokens

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "label_id": self.label_id,
            "tokens": self.tokens,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class Batch:
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.tokens = tokens

    def to(self, device):
        return Batch(
            input_ids=self.input_ids.to(device),
            input_mask=self.input_mask.to(device),
            segment_ids=self.segment_ids.to(device),
            label_ids=self.label_ids.to(device),
            tokens=self.tokens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, key):
        return Batch(
            input_ids=self.input_ids[key],
            input_mask=self.input_mask[key],
            segment_ids=self.segment_ids[key],
            label_ids=self.label_ids[key],
            tokens=self.tokens[key],
        )
