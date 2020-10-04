"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.transformer_boost import TransformerBoostEncoder
from onmt.encoders.transformer_multitask import TransformerMultiTask
from onmt.encoders.transformer_dense import TransformerDenseEncoder
from onmt.encoders.mlm_transformer import MLMTransformerEncoder
from onmt.encoders.transformer_boost_new import TransformerBoostEncoder as TransformerBoostEncoderNew


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder, "transformer_boost": TransformerBoostEncoder,
           "transformer_multi_task":TransformerMultiTask, "transformer_dense":TransformerDenseEncoder,
           'mlm_transformer': MLMTransformerEncoder, "transformer_boost_new": TransformerBoostEncoderNew}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc", "TransformerBoostEncoder", "TransformerMultiTask", "TransformerDenseEncoder",
           "MLMTransformerEncoder", "TransformerBoostEncoderNew"]
