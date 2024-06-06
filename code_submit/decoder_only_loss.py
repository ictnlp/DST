# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import pdb
import torch
from torch.nn.modules.module import Module
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn.functional as F


@dataclass
class LabelSmoothedCrossEntropyCriterionLMSiMTConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_lm_simt", dataclass=LabelSmoothedCrossEntropyCriterionLMSiMTConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True, update_num=0):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample['net_input']['src_tokens'].scatter(-1, sample['net_input']['src_lengths'].unsqueeze(1)-1, self.padding_idx)[:, :-1]
        #src_tokens = sample['net_input']['src_tokens']

        if self.training:
            mask_ratio = 0.3 + (1.0 - 0.3) * math.exp(-update_num / 30000.0)
        else:
            mask_ratio = 0.3
        # In the implementation, we remove the last <eos> tokens in the source sentence
        net_output = model([src_tokens, sample['net_input']['prev_output_tokens']], language_model_task=False, soft_attention=True, mask_ratio=mask_ratio)
        #net_output_lm = model(sample['net_input']['prev_output_tokens'], language_model_task=True)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        latency_loss = self.latency_loss(model, net_output[1]['merge_src_weight'], src_tokens, sample['net_input']['prev_output_tokens'], net_output[1]['training_metrix_signal'])
        limit_src_sum, limit_src_avg = self.attend_src_loss(model, net_output[1]['merge_src_weight'], src_tokens, sample['net_input']['prev_output_tokens'], net_output[1]['merge_signal'])
        #loss_lm, nll_loss_lm = self.compute_loss(model, net_output_lm, sample, reduce=reduce)
        #kl = F.kl_div(model.get_normalized_probs(net_output_lm, log_probs=True), model.get_normalized_probs(net_output, log_probs=False), reduction='sum')

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "limit_src_sum": limit_src_sum.data,
            "limit_src_avg": limit_src_avg.data,
            "latency_loss": latency_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if True:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        return loss + limit_src_avg + limit_src_sum + latency_loss, sample_size, logging_output

    def get_latency_targets(self, mian_diag, src_len, tgt_len):
        src_seq = torch.arange(1, src_len+1).cuda().unsqueeze(0)
        tgt_seq = torch.arange(1, tgt_len+1).cuda().unsqueeze(-1)
        
        latency_matrix = torch.abs(src_seq - tgt_seq * (float(src_len) / tgt_len) - (mian_diag - 1))
        tgt_index = tgt_len - int(mian_diag * (float(tgt_len) / src_len))
        latency_matrix[tgt_index:, -2:] = 0
        latency_matrix = (latency_matrix - 1).masked_fill((latency_matrix-1) <= 0, 0) / max(src_len, tgt_len)
        
        return latency_matrix.detach()
    
    def get_latency_targets_2(self, mian_diag, src_len, tgt_len):
        src_seq = torch.arange(1, src_len+1).cuda().unsqueeze(0)
        tgt_seq = torch.arange(1, tgt_len+1).cuda().unsqueeze(-1)
        
        latency_matrix = torch.abs(src_seq - tgt_seq * (float(src_len) / tgt_len) - mian_diag)
        latency_matrix = (latency_matrix - 1).masked_fill((latency_matrix-1) <= 0, 0) /  max(tgt_len, src_len)
        #latency_matrix = (latency_matrix - 1).masked_fill((latency_matrix-1) <= 0, 0)
        return latency_matrix.detach()
    
    def get_left_top_targets(self, mian_diag, src_len, tgt_len):
        src_seq = torch.arange(1, src_len+1).cuda().unsqueeze(0)
        tgt_seq = torch.arange(1, tgt_len+1).cuda().unsqueeze(-1)
        
        latency_matrix = src_seq - tgt_seq * (float(src_len) / tgt_len) - mian_diag
        latency_matrix = (latency_matrix - 1).masked_fill((latency_matrix-1) <= 0, 0) / max(src_len, tgt_len)
        
        return latency_matrix.detach()

    def right_top_latency(self, model, merge_src_delta, src_tokens, tgt_tokens, merge_training_metrix_signal):
        bsz=src_tokens.size(0)
        src_len=src_tokens.size(1)
        tgt_len=tgt_tokens.size(1)
        
        latency_matrix = self.get_left_top_targets(0, src_len, tgt_len)
        merge_src_delta = merge_src_delta.masked_fill(src_tokens.eq(self.padding_idx).unsqueeze(1).unsqueeze(1), 0)
        merge_src_delta = (latency_matrix.unsqueeze(0).unsqueeze(0) * merge_src_delta)
        
        merge_src_delta.masked_fill_(tgt_tokens.eq(self.padding_idx).unsqueeze(1).unsqueeze(-1), 0)
        
        merge_src_delta = merge_src_delta.contiguous().view(bsz, model.decoder.num_layers, -1).sum(dim=-1)
        merge_src_delta = merge_src_delta.sum(dim=-1) / model.decoder.num_layers
        
        return merge_src_delta.sum()

    def latency_loss(self, model, merge_src_delta, src_tokens, tgt_tokens, merge_training_metrix_signal):
        bsz=src_tokens.size(0)
        src_len=src_tokens.size(1)
        tgt_len=tgt_tokens.size(1)

        merge_src_delta = merge_src_delta.masked_fill(src_tokens.eq(self.padding_idx).unsqueeze(1).unsqueeze(1), 0)
        
        #latency_matrix = self.get_latency_targets((int(str(model.decoder.cfg._content['top_bound'])) + int(str(model.decoder.cfg._content['low_bound'])) - 1) / 2, src_len, tgt_len)
        latency_matrix = self.get_latency_targets_2(0, src_len, tgt_len)
        merge_src_delta = merge_src_delta / (merge_src_delta.sum(dim=-1, keepdim=True)+1e-9)
        merge_src_delta = (latency_matrix.unsqueeze(0).unsqueeze(0) * merge_src_delta)
        
        merge_src_delta.masked_fill_(tgt_tokens.eq(self.padding_idx).unsqueeze(1).unsqueeze(-1), 0)
        
        merge_src_delta = merge_src_delta.contiguous().view(bsz, model.decoder.num_layers, -1).sum(dim=-1)
        merge_src_delta = merge_src_delta.sum(dim=-1) / model.decoder.num_layers
        
        return merge_src_delta.sum()

    def attend_src_loss(self, model, attend_src_weights, src_tokens, tgt_tokens, signal):
        bsz=src_tokens.size(0)
        tgt_len=tgt_tokens.size(1)
        src_len=src_tokens.size(1)
        src_lengths=src_tokens.ne(self.padding_idx).sum(dim=-1).long().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(bsz, model.decoder.num_layers, tgt_len, 1)
        
        attend_src_weights_sum = torch.cumsum(attend_src_weights, dim=-1)
        attend_src_weights_sum = attend_src_weights_sum.contiguous().view(bsz, model.decoder.num_layers, tgt_len, src_len).gather(dim=-1, index=src_lengths-1).squeeze(-1)
        
        attend_src_weights = attend_src_weights.contiguous().view(bsz, model.decoder.num_layers, tgt_len, src_len)
        
        attend_src_weights_avg = torch.sum(attend_src_weights, dim=1, keepdim=True) / model.decoder.num_layers
        
        average_loss = (attend_src_weights - attend_src_weights_avg.detach()).pow(2).sum(dim=1) / model.decoder.num_layers
        average_loss.masked_fill_(tgt_tokens.eq(self.padding_idx).unsqueeze(-1), 0)
        average_loss.masked_fill_(src_tokens.eq(self.padding_idx).unsqueeze(1), 0)
        
        signal = torch.sum(signal, dim=1, keepdim=True) / model.decoder.num_layers
        sum_loss = (attend_src_weights_sum - signal.detach()).pow(2).sum(dim=1) / model.decoder.num_layers
        sum_loss.masked_fill_(tgt_tokens.eq(self.padding_idx), 0)
        
        return average_loss.sum(), sum_loss.sum()

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
    
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        mask = target.ne(self.padding_idx)
        if torch.sum(mask) > 0:
            n_correct = torch.sum(
                lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
            )
            total = torch.sum(mask)
        else:
            n_correct = torch.tensor(1).cuda()
            total = torch.tensor(1).cuda()
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        limit_src_sum = sum(log.get("limit_src_sum", 0) for log in logging_outputs)
        limit_src_avg = sum(log.get("limit_src_avg", 0) for log in logging_outputs)
        latency_loss_sum = sum(log.get("latency_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "limit_src_sum", limit_src_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "limit_src_avg", limit_src_avg / sample_size, sample_size, round=3
        )

        metrics.log_scalar(
            "latency_loss", latency_loss_sum / sample_size, sample_size, round=3
        )

        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
