import torch
import torch.nn as nn

from onmt.utils.loss import NMTLossCompute
from onmt.modules.util_class import Cast


class ExtraLossGenerator(nn.Module):
    def __init__(self, model_opt, decoder_embed_weight, tvocab_size):
        super(ExtraLossGenerator, self).__init__()
        if model_opt.generator_function == "sparsemax":
            self.gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            self.gen_func = nn.LogSoftmax(dim=-1)
        self.generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      tvocab_size),
            Cast(torch.float32),
            self.gen_func
        )
        if model_opt.share_decoder_embeddings:
            self.generator[0].weight = decoder_embed_weight
    
    def forward(self, model_output):
        if isinstance(model_output, dict) is not True:
            raise ValueError("Wrong input for ExtraLossGenerator.")
        output = model_output['dec_out']
        # fp32 output
        output = self.generator(output)

        return output

class NMTLossComputeWithExtraLoss(NMTLossCompute):
    """
    Standard NMT Loss Computation With Extra Loss in output.
    """

    def __init__(self, criterion, generator, normalization="sents",
                 lambda_coverage=0.0, lambda_align=0.0, multi_task=False,
                 extra_loss_weight=1.0):
        super(NMTLossComputeWithExtraLoss, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.lambda_align = lambda_align
        self.multi_task = multi_task
        self.extra_loss_weight = extra_loss_weight

    def _compute_loss(self, batch, output, target, std_attn=None,
                      coverage_attn=None, align_head=None, ref_align=None):
        if isinstance(output, dict):
            extra_loss = output['extra_loss']
            output = output['dec_out']
        else:
            extra_loss = torch.tensor(0.0)
        bottled_output = self._bottle(output)

        # here, from fp16 to fp32, if using half
        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        total_loss = loss + extra_loss.to(torch.float32) * self.extra_loss_weight

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(
                std_attn=std_attn, coverage_attn=coverage_attn)
            loss += coverage_loss
        if self.lambda_align != 0.0:
            if align_head.dtype != loss.dtype:  # Fix FP16
                align_head = align_head.to(loss.dtype)
            if ref_align.dtype != loss.dtype:
                ref_align = ref_align.to(loss.dtype)
            align_loss = self._compute_alignement_loss(
                align_head=align_head, ref_align=ref_align)
            loss += align_loss
        stats = self._stats(loss.clone(), scores, gtruth)
        stats.update_extra(extra_loss.detach().clone())

        return total_loss, stats

