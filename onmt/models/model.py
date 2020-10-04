""" Onmt NMT Model base class definition """
import torch.nn as nn
from onmt.encoders.encoder import EncoderExtraInputBase

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        if isinstance(src, dict):
            if "take_dict_input" not in vars(self.encoder).keys():
                src = src['src']
        
        if "enc_layer_grads" in kwargs.keys() and kwargs["enc_layer_grads"] is not None:
            if isinstance(src, dict) is True:
                src['enc_layer_grads'] = kwargs['enc_layer_grads']
            else:
                src = {
                    'src': src,
                    'enc_layer_grads': kwargs['enc_layer_grads']
                }
        if 'step' in kwargs:
            step = kwargs['step']
        else:
            step = 0
        enc_state, memory_bank, lengths = self.encoder(src, lengths, step=step)
        
        if isinstance(src, dict):
            src_dict = src
            src = src_dict['src']
        
        if isinstance(self.encoder, EncoderExtraInputBase):
            extra_loss = memory_bank['extra_loss']
            enc_layer_outs = memory_bank['encoder_layer_outputs']
            memory_bank = memory_bank['out']
        else:
            extra_loss = None

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        if extra_loss is not None:
            ret = {
                'dec_out': dec_out,
                'extra_loss': extra_loss,
            }
            if self.encoder.adv_gradient_boost:
                ret['enc_layer_outs'] = enc_layer_outs

        else:
            ret = dec_out

        return ret, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
    
    def get_init_params(self):
        named_params = self.named_parameters()
        ret = []
        filterd = []
        for key, val in named_params:
            if 'shuffle_matrix' not in key:
                ret.append(val)
            else:
                filterd.append(key)
        return ret

