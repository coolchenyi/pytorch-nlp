from nlp.nmt.onmt.modules.UtilClass import LayerNorm, Bottle, BottleLinear, \
    BottleLayerNorm, BottleSoftmax, Elementwise
from nlp.nmt.onmt.modules.Gate import context_gate_factory, ContextGate
from nlp.nmt.onmt.modules.GlobalAttention import GlobalAttention
from nlp.nmt.onmt.modules.ConvMultiStepAttention import ConvMultiStepAttention
from nlp.nmt.onmt.modules.ImageEncoder import ImageEncoder
from nlp.nmt.onmt.modules.CopyGenerator import CopyGenerator, CopyGeneratorLossCompute
from nlp.nmt.onmt.modules.StructuredAttention import MatrixTree
from nlp.nmt.onmt.modules.Transformer import \
   TransformerEncoder, TransformerDecoder, PositionwiseFeedForward
from nlp.nmt.onmt.modules.Conv2Conv import CNNEncoder, CNNDecoder
from nlp.nmt.onmt.modules.MultiHeadedAttn import MultiHeadedAttention
from nlp.nmt.onmt.modules.StackedRNN import StackedLSTM, StackedGRU
from nlp.nmt.onmt.modules.Embeddings import Embeddings, PositionalEncoding
from nlp.nmt.onmt.modules.WeightNorm import WeightNormConv2d

from nlp.nmt.onmt.Models import EncoderBase, MeanEncoder, StdRNNDecoder, \
    RNNDecoderBase, InputFeedRNNDecoder, RNNEncoder, NMTModel

from nlp.nmt.onmt.modules.SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from nlp.nmt.onmt.modules.SRU import SRU


# For flake8 compatibility.
__all__ = [EncoderBase, MeanEncoder, RNNDecoderBase, InputFeedRNNDecoder,
           RNNEncoder, NMTModel,
           StdRNNDecoder, ContextGate, GlobalAttention, ImageEncoder,
           PositionwiseFeedForward, PositionalEncoding,
           CopyGenerator, MultiHeadedAttention,
           LayerNorm, Bottle, BottleLinear, BottleLayerNorm, BottleSoftmax,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU,
           context_gate_factory, CopyGeneratorLossCompute]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])