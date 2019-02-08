import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertConfig, BertLayer


class Masker(nn.Module):
    def __init__(self, vocab_size, original_hidden_size, num_layers, tau=1):
        super().__init__()
        self.bert_layer = BertLayer(BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=original_hidden_size * num_layers,
        ))
        self.linear_layer = nn.Linear(original_hidden_size * num_layers, 1)
        self.log_sigmoid = nn.LogSigmoid()
        self.tau = tau

    def forward(self, x, attention_mask, gumbel_softmax=True, tau=None):
        extended_attention_mask = self.convert_mask(attention_mask)
        h = self.bert_layer(x, extended_attention_mask)
        h = self.linear_layer(h)
        log_probs = self.log_sigmoid(h).squeeze(dim=2)

        if gumbel_softmax:
            tau = self.tau if tau is None else tau
            return F.gumbel_softmax(log_probs, tau=tau)
        else:
            return log_probs

    def convert_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
