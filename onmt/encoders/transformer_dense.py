"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

class TransformerDenseBlock(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """ 
    def __init__(self, d_model, d_base, growth_rate, num_layers, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0, self_att_orthogonal=False):
        super(TransformerDenseBlock, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers = num_layers    
        self.layers = nn.ModuleList([TransformerEncoderLayer(
                                            d_base, heads, d_ff, dropout, attention_dropout,
                                            max_relative_positions=max_relative_positions, self_att_orthogonal=self_att_orthogonal)
                                    for i in range(num_layers)])
        self.pre_transform = nn.ModuleList([nn.Linear(d_model+growth_rate*i, d_base) for i in range(num_layers)])
        self.post_transform = nn.ModuleList([nn.Linear(d_base, growth_rate) for i in range(num_layers)])
        self.post_ln = nn.ModuleList([nn.LayerNorm(growth_rate, eps=1e-6) for i in range(num_layers)])
        # self.relu = nn.Relu()
    
    def forward(self, inputs, mask):
        concat_layers = [inputs]
        for i in range(self.num_layers):
            # bz, length, d_model + i*growth_rate
            x = torch.cat(concat_layers, dim=-1)
            # bz, length, d_base
            x = self.pre_transform[i](x)
            # x = self.relu(x)
            # bz, length, d_base
            x = self.layers[i](x, mask)
            # bz, length, growth
            x = self.post_transform[i](x)
            x = self.post_ln[i](x)
            x = F.relu(x)
            concat_layers.append(x)
        return torch.cat(concat_layers, dim=-1)

class Transition(nn.Module):
    def __init__(self, d_in, d_out):
        super(Transition, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        # self.relu = nn.Relu()
        self.ln = nn.LayerNorm(d_out, eps=1e-6)
    
    def forward(self, inputs):
        out = inputs
        out = self.linear(out)
        out = self.ln(out)
        out = F.relu(out)
        return out

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0, self_att_orthogonal=False):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions,
            orthogonal=self_att_orthogonal)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerDenseEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions, self_att_orthogonal,
                 dense_growth_rate, dense_block_dim, num_dense_layers):
        super(TransformerDenseEncoder, self).__init__()

        self.embeddings = embeddings
        self.dense_growth_rate = dense_growth_rate
        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [TransformerDenseBlock
                (d_model, dense_block_dim, dense_growth_rate, num_dense_layers,
                    heads, d_ff, dropout, attention_dropout,
                    max_relative_positions=max_relative_positions, self_att_orthogonal=self_att_orthogonal)
             for i in range(num_layers)])   
        self.transitions = nn.ModuleList([
            Transition(d_model+num_dense_layers*dense_growth_rate, d_model) for i in range(num_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.self_att_orthogonal,
            opt.dense_growth_rate,
            opt.dense_block_dim,
            opt.num_dense_layers)


    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for i, layer in enumerate(self.transformer):
            x = out
            out = layer(out, mask)
            out = self.transitions[i](out)
            # add skip connections
            out = x + out
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
