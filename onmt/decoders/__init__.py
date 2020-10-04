"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.decoders.transformer_multitask import TransformerDecoderMultiTask
from onmt.decoders.transformer_causal import TransformerCausal

str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder, "transformer_multi_task":TransformerDecoderMultiTask,
           "transformer_causal":TransformerCausal}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", "TransformerDecoderMultiTask", "TransformerCausal"]
