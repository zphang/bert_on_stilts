class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, glue_label_id, lm_label_ids, tokens):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.glue_label_id = glue_label_id
        self.lm_label_ids = lm_label_ids
        self.tokens = tokens

    def new(self, **new_kwargs):
        kwargs = {
            "guid": self.guid,
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "glue_label_id": self.glue_label_id,
            "lm_label_ids": self.lm_label_ids,
            "tokens": self.tokens,
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class Batch:
    def __init__(self, input_ids, input_mask, segment_ids, glue_label_ids, lm_label_ids, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.glue_label_ids = glue_label_ids
        self.lm_label_ids = lm_label_ids
        self.tokens = tokens

    def to(self, device):
        return Batch(
            input_ids=self.input_ids.to(device),
            input_mask=self.input_mask.to(device),
            segment_ids=self.segment_ids.to(device),
            glue_label_ids=self.glue_label_ids.to(device),
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
            glue_label_ids=self.glue_label_ids[key],
            lm_label_ids=self.lm_label_ids[key],
            tokens=self.tokens[key],
        )
