"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

import pdb

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

    def forward(self, inputs, mask, addi_pos):
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
                                    mask=mask, attn_type="self", addi_mask=addi_mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class MLMTransformerEncoder(EncoderBase):
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
                 attention_dropout, embeddings, max_relative_positions, self_att_orthogonal):
        super(MLMTransformerEncoder, self).__init__()

        self.embeddings = embeddings
        print("Make sure the last token in src embeddings is <mask>.")
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions, self_att_orthogonal=self_att_orthogonal)
             for i in range(num_layers)])
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
            opt.self_att_orthogonal)


    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``


        Returns:
            (FloatTensor, FloatTensor):

            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
        """

        self._check_args(src, lengths)
        
        # [bz, len]
        src = src.transpose(0, 1)
        # random select 15%
        batch_size, length= src.shape[:2]
        addi_len = int(float(length) * 0.15)
        # sampling position, [bz, addi_len]
        pos = torch.randint(high=length, size=(batch_size, addi_len), device=src.device)
        # [bz, addi_len, 1]
        addi_src = torch.gather(src.squeeze(), 1, pos).unsqueeze(-1)

        # 10% un-mod, 10% random, 80% mask
        # sample uniform distribution, [bz, addi_len, 1]
        _prob = torch.rand([batch_size, addi_len, 1], device=src.device)
        un_mod = (_prob < 0.1).float()
        rand_repl = (_prob < 0.2).float() - un_mod
        mask = (_prob >= 0.2).float()
        assert int((un_mod + rand_repl + mask).sum()) == batch_size * addi_len
        assert int(un_mod.sum() + rand_repl.sum() + mask.sum()) == batch_size * addi_len
        vocab_size = self.embeddings.emb_luts[0].weight.shape[0]
        # we set the mask to be the last token in the vocab.
        mask_val = vocab_size - 1
        rand_repl_val = torch.randint(high=vocab_size, size=([batch_size, addi_len, 1]), device=src.device)
        # merge 3 categories into one. [bz, addi_len, 1]
        addi_src = addi_src.float() * un_mod + mask * mask_val + rand_repl_val.float() * rand_repl
        # convert back to long tensor
        addi_src = addi_src.long()
        # [bz, len + addi_len, 1]
        src = torch.cat([src, addi_src], dim=1)

        # TODO: relative position is not considered yet.
        # [bz, len]
        step = torch.arange(length, device=src.device).unsqueeze(0).repeat(batch_size, 1)
        # [len, bz], prepare for position encoding.
        new_step = torch.cat([step, pos], dim=1).transpose(0, 1)
        # [len + addi_len, bz, dim]
        emb = self.embeddings(src.transpose(0, 1), step=new_step)

        out = emb.transpose(0, 1).contiguous()
        # [bz, 1, len]
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # modify the mask.
        addi_mask = torch.zeros([batch_size, 1, addi_len], device=mask.device, dtype=mask.dtype)
        # [bz, 1, len + addi_len]
        mask = torch.cat([mask, addi_mask], dim=-1)
        self_addi_mask = self.make_additional_mask(pos, length)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            # need to pass in extra position information.
            out = layer(out, mask, self_addi_mask)
        out = self.layer_norm(out)
        # split it out.


        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
    
    def make_additional_mask(self, pos, ori_len):
        """ Create additional mask used in self-attention.
            Arguments:
                pos: indicate additional sampled positions. 
                ``(batch_size, addi_len)``

            Return: 
                mask: (batch_size, ori_len+addi_len, ori_len+total_len)
        """
        batch_size, addi_len = pos.shape[:2]
        total = addi_len + ori_len
        # [bz, total_len, total_len]
        mask = torch.cat([torch.zeros([batch_size, total_len, ori_len]), 
                          torch.ones([batch_size, total, addi_len])], dim=-1).to(pos.device)
        


