"""
Implementation of "Attention is All You Need"
"""

import pdb
import math
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.encoder import EncoderBase, EncoderExtraInputBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
from onmt.utils.logging import logger


class TransformerEncoderBoostLayer(nn.Module):
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
                 max_relative_positions=0, num_boost=4, learnable_weights=True, 
                 boost_type='continuous', main_stream=False, boost_drop_rate=0.1, 
                 boost_dropout_diff=0.0,boost_with_ffn=False, boost_str='', 
                 boost_gating=False, mask_pos_type=[], self_att_merge_layer=False,
                 adv_bias_step=0.0, shuffle_merge=False, shuffle_merge_type="sum",
                 adv_gradient_boost=False,
                 adv_gradient_boost_step=0.01, adv_gradient_boost_func='mse',
                 adv_gradient_boost_no_ce=False, gradient_boost_scale=1.0,
                 boost_adv_method_list=[]):
                 
        super(TransformerEncoderBoostLayer, self).__init__()

        self.num_boost = num_boost
        self.boost_type = boost_type
        self.main_stream = main_stream
        self.boost_drop_rate = boost_drop_rate
        self.boost_with_ffn = boost_with_ffn
        self.use_adv = True if self.boost_type == 'adv' else False
        self.a_num = num_boost
        # self.use_dropout_diff = True if boost_dropout_diff != 0.0 else False
        self.use_dropout_diff = False
        self.d_num = num_boost
        self.use_mask = True if self.boost_type in {'continuous', 'continuous_comp', 'random', 'pos'} else False
        # overwrite params based on boost_str
        self.boost_gating = boost_gating
        self.mask_pos_type = mask_pos_type
        # init postag params
        self.use_postag = False
        self.p_num = 0
        self._parse_boost_str(boost_str)
        # whether to use self-att to merge each path's output
        self.use_self_att_merge_layer = self_att_merge_layer
        
        self.adv_bias_step = adv_bias_step
        self.shuffle_merge = shuffle_merge
        self.shuffle_merge_type = shuffle_merge_type

        self.adv_gradient_boost = adv_gradient_boost
        self.adv_gradient_boost_step = adv_gradient_boost_step
        self.adv_gradient_boost_func = adv_gradient_boost_func
        self.adv_gradient_boost_no_ce = adv_gradient_boost_no_ce

        self.gradient_boost_scale = gradient_boost_scale 

        # compute dropout list
        if not self.use_dropout_diff:
            dropout_list = [dropout for i in range(self.num_boost)]
        else:
            dropout_diffs = [boost_dropout_diff * i - float(self.d_num)/2 * boost_dropout_diff for i in range(self.d_num)]
            dropout_list = [dropout + dropout_diffs[i] for i in range(self.d_num)] + [dropout for i in range(self.num_boost - self.d_num)]
        self.dropout_list = dropout_list
        print("Boost dropout list: {}".format(dropout_list))
        assert max(dropout_list) <= 1.0 and min(dropout_list) >= 0.0
        
        # list of self-attention module
        self.self_attn_list = [ MultiHeadedAttention(
                                    heads, d_model, dropout=attention_dropout,
                                    max_relative_positions=max_relative_positions)
                                for n in range(self.num_boost) ]
        self.self_attn_list = nn.ModuleList(self.self_attn_list)

        if self.main_stream:
            # main stream for self-attention
            self.main_self_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout,
                                                       max_relative_positions=max_relative_positions)
        
        if self.use_self_att_merge_layer:
            # keep the default setting for self-attention layer.
            self.att_merge_layer = MultiHeadedAttention(
                heads, d_model, dropout=attention_dropout,
                max_relative_positions=max_relative_positions)
            self.merge_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # convert all ones to 1/N
        weights_init = torch.ones(self.num_boost, dtype=torch.float32) / self.num_boost
        # if learnable_weights is True:
        self.weights = nn.Parameter(weights_init, requires_grad=learnable_weights)
        # else:
            # self.weights = weights_init
        
        if self.boost_with_ffn:
            feed_forward_list = [ PositionwiseFeedForward(d_model, d_ff, dropout_list[i]) for i in range(self.num_boost) ]
            self.feed_forward = nn.ModuleList(feed_forward_list)
        else:
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norms = [nn.LayerNorm(d_model, eps=1e-6) for _ in range(self.num_boost)]
        self.layer_norms = nn.ModuleList(self.layer_norms)
        self.dropout = nn.Dropout(dropout)
        # TODO: Functions for drop_rate is not implemented yet.

        if self.shuffle_merge:
            shuffle_matrix = torch.abs(torch.randn(self.num_boost, self.num_boost))
            self.shuffle_matrix = nn.Parameter(shuffle_matrix)
            self.merge_weights = torch.ones((self.num_boost-1,), dtype=torch.float32, requires_grad=False)

        if self.use_adv or self.use_postag is True:
            # permutation of max position range.
            self.max_perm = 3
            self.max_exchange = 3
            if not boost_adv_method_list:
                all_adv_methods = ['swap', 'reorder', 'delete', 'mask']
            else:
                all_adv_methods = boost_adv_method_list
            assert self.a_num <= len(all_adv_methods)
            self.activate_methods = all_adv_methods[:self.a_num]
            # create mask tensor
            if "mask" in self.activate_methods or self.use_postag is True:
                mask_tensor = torch.empty(d_model)
                torch.nn.init.normal_(mask_tensor, std=1.0/math.sqrt(d_model))
                self.mask_tensor = nn.Parameter(mask_tensor)

            print('Activated adversarial methods: {}'.format(self.activate_methods))
        
        if self.use_postag:
            assert len(self.mask_pos_type) == self.p_num

        if self.adv_gradient_boost is True:
            if adv_gradient_boost_func == 'mse':
                self.mse = nn.MSELoss(reduction='none')
            elif adv_gradient_boost_func == 'cos':
                self.cos_sim = nn.CosineSimilarity(dim=2)
            elif adv_gradient_boost_func == 'l1':
                self.l1 = nn.L1Loss(reduction='none')
            else:
                raise ValueError()
        
        self.keep_adv_gradient = False
        self.adv_gradient_value = 'moving_average'
        
        return
    
    def _parse_boost_str(self, boost_str):
        # parse with boost_str --> for all combinations
        # boost str format:  str, e.g., "a2 d4"
        keys = defaultdict()
        boost_options = boost_str.split()
        for option in boost_options:
            k = option[0]
            n = int(option[1:])
            keys[k] = n
        self._update_params(keys)
        return
    
    def _update_params(self, dict):
        _tot = 0
        for key, val in dict.items():
            if key == 'a':
                # set adversarial
                self.use_adv = True
                self.a_num = val
                _tot += val
            elif key == 'd':
                # set dropout diff
                self.use_dropout_diff = True
                self.d_num = val
                _tot += val
            elif key == 'p':
                self.use_postag = True
                self.p_num = val
                _tot += val
        # update total number of boost
        self.num_boost = _tot
        return

        
    def forward(self, inputs, mask, enc_layer_grads=None, **kwargs):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        # check inputs is dict or embeddings
        src_pos = kwargs.get('src_pos')
        
        if self.boost_with_ffn is True:
            # input_norm = self.layer_norm(inputs)
            input_norms = [self.layer_norms[i](inputs) for i in range(self.num_boost)]
            if self.training:
                input_norm_list, mask_list = self.create_input_norms(input_norms, mask, src_pos)
            else:
                if src_pos is not None:
                    input_norm_list, mask_list = self.create_input_norms(input_norms, mask, src_pos)
                else:
                    input_norm_list = input_norms
                    mask_list = [torch.ones_like(mask).transpose(1,2) for i in range(self.num_boost)]

            mask_list = [(1.0 - item.transpose(1, 2).long()) for item in mask_list]
            
            '''
            steams = [torch.cuda.Stream() for _ in range(self.num_boost)]
            torch.cuda.synchronize()
            outs = [None] * self.num_boost
            # import pdb; pdb.set_trace()
            for i, s in enumerate(steams):
                with torch.cuda.device(i):
                    # print(torch.cuda.current_device())
                    with torch.cuda.stream(s):
                        self_attn_out = self.self_attn_list[i](input_norm_list[i], input_norm_list[i], 
                                        input_norm_list[i], mask=(mask.long()+mask_list[i]).gt(0.0).to(mask.dtype), 
                                        attn_type="self")
                        out = F.dropout(self_attn_out[0], p=self.dropout_list[i], training=self.training) + inputs
                        out = self.feed_forward[i](out)
                        outs[i] = out
            torch.cuda.synchronize()
            '''

            self_attn_outs = [self.self_attn_list[i](input_norm_list[i], input_norm_list[i], 
                                                    input_norm_list[i], mask=(mask.long()+mask_list[i]).gt(0.0).to(mask.dtype), attn_type="self")
                            for i in range(self.num_boost)]
            # bz, length, D
            outs = [ F.dropout(self_attn_outs[i][0], p=self.dropout_list[i], training=self.training)
                    + inputs for i in range(self.num_boost) ]
            del self_attn_outs
            # bz, length, D
            outs = [ self.feed_forward[i](outs[i]) for i in range(self.num_boost) ]
            # bz, length, D, num_boost
            new_outs = torch.stack(outs, dim=-1)
            del outs

            ret = dict()
            # call merge layer
            return_outs = self.adv_gradient_boost and enc_layer_grads is None
            merge_outs = self.merge_layer(None, new_outs, enc_layer_grads, mask, return_outs=return_outs)
            if return_outs:
                out, outs, extra_loss = merge_outs
                ret['outs'] = outs
            else:
                out, extra_loss = merge_outs

            if self.shuffle_merge:
                extra_loss = self._compute_shuffle_matrix_regularization(mask) + extra_loss

            ret['out'] = out
            ret['extra_loss'] = extra_loss
            # first pass
            return ret
        else:
            raise ValueError("Use boost with ffn")

    def merge_layer(self, input_norm, outs, enc_layer_grads, mask, return_outs=False):
        """ Inputs:
                input_norm, [bz, length, dim]
                outs, [bz, length, dim, num_boost]
        """
        extra_loss = 0.0

        # enable shuffle merge and boost alg
        if self.shuffle_merge:
            self._update_shuffle_matrix()
            # shuffle the outputs, [bz, length, dim, num_boost]
            outs = torch.matmul(outs, self.shuffle_matrix)
            # if self.td_outs = outs
            merge_outs = self._shuffle_merge(outs)

            if self.adv_gradient_boost:
                merge_outs.retain_grad()
                extra_loss = extra_loss + self._compute_gradient_boost_loss(outs, merge_outs, enc_layer_grads, mask)
            
            outs = merge_outs

        # determine the weights for each path output 
        if not self.boost_gating:
            # weights: [num_boost]
            weights = self.weights
            # in the second forward pass.
            if weights.grad is not None and self.adv_bias_step != 0 and self.adv_gradient_boost:
                weights = self.adv_bias_step * weights.grad.data + weights
        else:
            '''
            # [bz, d_model]
            avg = input_norm.mean(dim=1)
            weights = self.g_linear(avg)
            if self.training:
                bz = input_norm.shape[0]
                # sample from standard normal distribution.
                mean = torch.zeros([bz, self.num_boost], dtype=inputs.dtype, device=input_norm.device)
                std = torch.ones([bz, self.num_boost], dtype=inputs.dtype, device=input_norm.device)
                # [bz, num_boost]
                weights = weights + torch.normal(mean=mean, std=std) * self.softplus(self.n_linear(avg))
            # [bz, 1, 1, num_boost]
            weights = weights.unsqueeze(1).unsqueeze(1)
            '''
            raise ValueError('Wrong parameters')
        
        out = (outs * weights).sum(dim=-1)
        if not return_outs:
            return out, extra_loss
        else:
            return out, outs, extra_loss

    def _shuffle_merge(self, outs):
        if self.shuffle_merge_type == 'sum':
            new_outs = []
            for i in range(self.num_boost):
                if i == 0:
                    new_outs.append(outs[:,:,:,i])
                else:
                    # accum merge
                    # NOTE: add merge weights, haven't implemented on avg and half yet.
                    cur_out = outs[:, :, :, i]
                    if self.adv_gradient_boost_no_ce is True:
                        cur_out = cur_out.detach()
                    new_outs.append(new_outs[-1] + self.merge_weights[i-1] * cur_out)
        elif self.shuffle_merge_type == 'avg':
            new_outs = []
            for i in range(self.num_boost):
                if i == 0:
                    new_outs.append(outs[:,:,:,i])
                else:
                    # accum merge
                    new_outs.append((new_outs[-1] + outs[:, :, :, i]))
            # take average
            new_outs = [item / (i+1) for i, item in enumerate(new_outs)]
        elif self.shuffle_merge_type == 'half':
            new_outs = []
            for i in range(self.num_boost):
                if i == 0:
                    new_outs.append(outs[:,:,:,i])
                else:
                    # accum merge
                    new_outs.append((new_outs[-1] * 0.5 + outs[:, :, :, i] * 0.5))
        return torch.stack(new_outs, dim=-1)
    
    def _update_shuffle_matrix(self):
        # keep it >= 0
        shuffle_matrix = F.relu(self.shuffle_matrix)
        # unit normalize over rows and columns
        shuffle_matrix = shuffle_matrix / shuffle_matrix.sum(dim=1, keepdim=True)
        shuffle_matrix = shuffle_matrix / shuffle_matrix.sum(dim=0, keepdim=True)
        self.shuffle_matrix.data = shuffle_matrix
        return
    
    def _compute_shuffle_matrix_regularization(self, mask):
        """ Return:
                penalty_loss : float32 
        """
        shuffle_matrix = self.shuffle_matrix.to(torch.float32)
        # [1, num_boost]
        col_norm = shuffle_matrix.norm(dim=0, keepdim=True)
        # [num_boost, 1]
        row_norm = shuffle_matrix.norm(dim=1, keepdim=True)

        row_sum = torch.sum(torch.abs(shuffle_matrix), 1, keepdim=True)
        col_sum = torch.sum(torch.abs(shuffle_matrix), 0, keepdim=True)
        # penalty_loss = torch.sum((self.shuffle_matrix), dtype=self.shuffle_matrix.dtype)
        penalty_loss = torch.sum((row_sum - row_norm)) + torch.sum((col_sum - col_norm))
        # penalty_loss = penalty_loss * (1.0-mask.to(penalty_loss.dtype)).sum()

        return penalty_loss
    
    def _compute_gradient_boost_loss(self, outs, merge_outs, enc_layer_grads, mask):
        """Compute gradient boost loss based on each path outputs.
        
        Arguments:
            outs {Tensor, [bz, length, dim, num_boost]} -- outputs for each path, before merge.
            merge_outs {Tensor, [bz, length, dim, num_boost]} -- outputs for each path, after merge.
            enc_layer_grads {Tensor, [bz, length, dim, num_boost] or None} -- Gradients for merge_outs. 
                None in the first forward pass.
            mask {Tensor, [bz, src_len]} -- src_pad_mask.
        
        Returns:
            loss {Tensor, []}
        """
        # outs, [bz, length, dim, num_boost]
        if enc_layer_grads is None:
            # skip in the first pass
            return 0.0
        # move one step forward, follow negative gradient
        gd_targets = - enc_layer_grads * self.adv_gradient_boost_step
        # shift right
        gd_targets = gd_targets[:,:,:,:-1]
        outs = outs[:,:,:,1:]
        assert gd_targets.size() == outs.size()
        # [bz, length]
        mask = (1.0-mask.to(outs.dtype)).squeeze(1)
        if self.adv_gradient_boost_func == 'mse':
            # [bz, length, dim, num_boost-1]
            mse_loss = self.mse(outs, gd_targets.detach())
            # [bz, length]
            mse_loss = mse_loss.mean(dim=-1).mean(dim=-1) * mask
            loss = mse_loss.sum()
        elif self.adv_gradient_boost_func == 'cos':
            # NOTE: not debug yet.
            # [bz, length, num_boost-1]
            cos_sim_loss = self.cos_sim(outs, gd_targets.detach())
            # [bz, length]
            cos_sim_loss = cos_sim_loss.mean(dim=-1) * mask
            loss = cos_sim_loss.sum()
            # raise NotImplementedError("Not impl yet.")
        elif self.adv_gradient_boost_func == 'l1':
            batch, length, dim, num_boost = outs.shape
            # [-1, dim]
            gd_targets = gd_targets.transpose(2, 3).contiguous().view(-1, dim)
            outs = outs.transpose(2, 3).contiguous().view(-1, dim)
            l1_loss = self.l1(outs, gd_targets.detach())
            l1_loss = l1_loss.contiguous().view(batch, length, num_boost, dim)
            l1_loss = l1_loss.mean(dim=-1).mean(dim=-1) * mask
            loss = l1_loss.sum()
        else:
            raise ValueError("Wrong parameter for adv_gradient_boost_func.")
        loss = loss * self.gradient_boost_scale
        return loss

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

    def _get_postag_mask(self, src_pos, key):
        """ Input: 
                src_pos, shape: [length, bz, 1]
            Return:
                mask, shape: [bz, length, 1]
        """
        # src_pos = src_pos.squeeze(-1)
        if key == 'DE':
            right = src_pos < 9
            left = src_pos < 6
            mask = left + right
        elif key == 'PN':
            mask = ~(src_pos == 25)
        elif key == 'V':
            right = src_pos > 33
            left = src_pos < 31
            mask = left + right
        elif key == 'N':
            right = src_pos > 21
            left = src_pos < 19
            mask = left + right
        elif key == 'ADV':
            mask = ~(src_pos == 0)
        elif key == 'ADJ':
            mask = ~(src_pos == 30)
        else:
            raise ValueError('Wrong postag mask key.')
        mask = mask.transpose(0, 1).float().to(src_pos.device)
        return mask

    def create_input_norms(self, input_norm_list, mask, src_pos=None):
        """
            inputs: input_norm_list, list of tensor, shape: [bz, length, dim]
                    mask, shape: [bz, 1, length], 1 denotes need for mask.
                    src_pos, shape: [length, bz, 1] or None.

            return: ret, list of tensor, shape like input_norm
                    mask_list, list of tensor, shape: [batch_size, length, 1], 0 denotes need for mask.
        """
        ret = []
        mask_list = []
        cnt_boost = 0

        '''
        if self.use_mask is True:
            # TODO: cannot be directly used. need modifications.
            if self.boost_type.startswith('continuous') or self.boost_type == 'continous':
                # [bz]
                lengths = (~mask).sum(dim=-1).squeeze()
                batch_size, max_len = mask.shape[0], mask.shape[-1]
                # [bz]
                each_span = (lengths-1) / (self.num_boost + 1) + 1
                # list of [bz, ]
                left_bound = [(each_span * item).unsqueeze(1) for item in range(self.num_boost)]
                # list of [bz, ]
                right_bound = [(each_span * (item+1)).unsqueeze(1) for item in range(self.num_boost)]

                for i in range(self.num_boost):
                    tmp = torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).repeat(batch_size, 1)
                    # bz, length
                    left_mask = tmp.ge(left_bound[i])
                    # bz, length
                    right_mask = tmp.lt(right_bound[i])
                    # bz, length, 1
                    tmp_mask = (left_mask + right_mask).eq(2.0).float().unsqueeze(-1)
                    if self.boost_type == 'continuous_comp':
                        tmp_mask = 1.0 - tmp_mask
                    ret.append(tmp_mask * input_norm)
                    mask_list.append(tmp_mask)
                return ret, mask_list
            
            elif self.boost_type == 'random':
                batch_size, max_len = input_norm.shape[0], input_norm.shape[1]
                ret = []
                for i in range(self.num_boost):
                    p = (1.0 / self.num_boost) * torch.ones([batch_size, max_len], device=input_norm.device).float()
                    # [bz, max_len]
                    mask_tensor = torch.bernoulli(p).unsqueeze(-1)
                    ret.append(input_norm * mask_tensor)
                return ret
        '''

        if self.use_dropout_diff is True:
            ret += [input_norm_list[i+cnt_boost] for i in range(self.d_num)]
            mask_list += [torch.ones_like(mask).transpose(1,2) for i in range(self.d_num)]
            cnt_boost += self.d_num
        
        if self.use_postag is True:
            assert src_pos is not None
            # src_pos, shape: [length, bz, 1]
            pos_ret = []
            for i, key in enumerate(self.mask_pos_type):
                # [bz, length, 1]
                pos_mask = self._get_postag_mask(src_pos, key)
                masked_input = input_norm_list[i+cnt_boost] * pos_mask + (1.0 - pos_mask) * self.mask_tensor
                pos_ret.append(masked_input)
                # masked_inputs = input_norm * bernoulli_mask + (1.0 - bernoulli_mask) * self.mask_tensor
            ret += pos_ret
            mask_list += [torch.ones_like(mask).transpose(1,2) for i in range(self.p_num)]
            cnt_boost += self.p_num

        if self.use_adv is True:
            if self.training is False:
                # if in test mode, skip all operations in adversarial.
                ret += [input_norm_list[i+cnt_boost] for i in range(self.a_num)]
                mask_list += [torch.ones_like(mask).transpose(1,2) for i in range(self.a_num)]
                cnt_boost += self.a_num
            else:
                length = (1.0 - mask.long()).sum(-1).squeeze(-1)
                # length = (1.0 - mask.to(input_norm.dtype)).sum(-1).squeeze(-1)
                input_shape = input_norm_list[0].shape
                batch_size, max_len = input_shape[:2]
                d_model = input_shape[-1]
                batch_arange_tensor = torch.arange(batch_size, device=input_norm_list[0].device)
                
                for k in range(self.a_num):
                    input_norm = input_norm_list[k+cnt_boost]
                    # reorder
                    if 'reorder' == self.activate_methods[k]:
                        new_input_norm = input_norm.clone()
                        for i in range(batch_size):
                            # choose a position, in cpu
                            pos = torch.randint(high=length[i], size=(1,))
                            perm = torch.randperm(n=self.max_perm, device=input_norm.device)
                            # TODO: try to find way to solve corner case. skip for now.
                            if int(pos + self.max_perm) > length[i]:
                                continue
                            # to gpu
                            pos = pos.to(input_norm.device)
                            new_input_norm[i][pos:pos+self.max_perm] = torch.index_select(input_norm[i], dim=0, index=perm+pos)
                        ret.append(new_input_norm)
                        mask_list.append(torch.ones_like(mask).transpose(1,2))
                    
                    if 'swap' == self.activate_methods[k]:
                        # swap. Note, this should only be useful when applying relative position.
                        # approximately random sampling.
                        pos1 = torch.randint(high=max_len, size=(batch_size,), device=input_norm.device)
                        # keep it in range.
                        pos1 = torch.min(pos1, length)
                        # sample diff
                        pos1_diff = torch.randint(low=-self.max_exchange, high=self.max_exchange, size=(batch_size,),
                                                device=pos1.device, dtype=pos1.dtype)
                        pos2 = pos1_diff + pos1
                        # keep it in range
                        pos2 = torch.min(torch.max(pos2, torch.tensor(0, device=pos2.device)), length-1)
                        
                        new_input_norm = input_norm.clone()
                        # get swap feature
                        a = input_norm[batch_arange_tensor, pos1]
                        b = input_norm[batch_arange_tensor, pos2]
                        # swap
                        new_input_norm[batch_arange_tensor, pos1] = b
                        new_input_norm[batch_arange_tensor, pos2] = a
                        ret.append(new_input_norm)
                        mask_list.append(torch.ones_like(mask).transpose(1,2))

                    if 'delete' == self.activate_methods[k]:
                        # delete
                        # [bz, 1]
                        pos = torch.randint(high=max_len, size=(batch_size,), device=input_norm.device)
                        new_input_norm = input_norm.clone()
                        # zero out the deleted position.
                        new_input_norm[batch_arange_tensor, pos] = 0.
                        # [bz, length, 1]
                        tmp = torch.arange(0, max_len, device=input_norm.device).long().unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
                        # [bz, length, 1] 
                        left_mask = (tmp < pos.view(-1, 1, 1)).float()
                        # [bz, length, 1] 
                        right_mask = 1.0 - left_mask
                        # compute left content
                        left_content = new_input_norm * left_mask
                        # compute right content
                        right_content = new_input_norm * right_mask
                        # shift left, at least one 0 on the left
                        # [bz, length, d_model]
                        right_content = torch.cat([right_content[:, 1:], torch.zeros(batch_size, 1, d_model, device=right_content.device)],
                                                dim=1)
                        new_input_norm = left_content + right_content
                        # bz, length, 1
                        delete_mask = torch.ones_like(mask).transpose(1,2)
                        delete_mask[:, -1] = 0.0
                        mask_list.append(delete_mask)
                        ret.append(new_input_norm)

                    if 'mask' == self.activate_methods[k]:
                        # mask
                        # bz, length, 1
                        _prob = torch.empty(input_shape[:-1], 
                                            device=input_norm.device).fill_(0.9).to(input_norm.dtype).unsqueeze(-1)
                        # bz, length, 1
                        bernoulli_mask = torch.bernoulli(_prob)
                        # mask by trainable mask tensor.
                        masked_inputs = input_norm * bernoulli_mask + (1.0 - bernoulli_mask) * self.mask_tensor
                        ret.append(masked_inputs)
                        mask_list.append(torch.ones_like(mask).transpose(1,2))
                cnt_boost += self.a_num
        
        assert cnt_boost == self.num_boost
        assert len(ret) == self.num_boost
        assert len(mask_list) == self.num_boost
        return ret, mask_list

class TransformerBoostEncoder(EncoderExtraInputBase):
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
                 attention_dropout, embeddings, max_relative_positions, 
                 num_boost, boost_type, boost_main, boost_drop_rate, 
                 boost_dropout_diff, boost_with_ffn, boost_str, boost_gating,
                 mask_pos_type, disable_learnable_weights, self_att_merge_layer,
                 adv_bias_step, adv_bias_layer, shuffle_merge, shuffle_merge_type,
                 adv_gradient_boost, adv_gradient_boost_step, adv_gradient_boost_func,
                 adv_gradient_boost_no_ce, gradient_boost_scale, boost_adv_method_list):
        super(TransformerBoostEncoder, self).__init__()
        self.embeddings = embeddings

        # compute adv bias layers
        enable_adv_bias_layers = set(list(range(num_layers))[-adv_bias_layer:]) if adv_bias_layer != 0 else set()
        self.transformer = nn.ModuleList(
            [TransformerEncoderBoostLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions,
                num_boost=num_boost, learnable_weights=(not disable_learnable_weights),
                boost_type=boost_type, 
                main_stream=boost_main, boost_drop_rate=boost_drop_rate,
                boost_dropout_diff=boost_dropout_diff,
                boost_with_ffn=boost_with_ffn, boost_str=boost_str,
                boost_gating=boost_gating, mask_pos_type=mask_pos_type,
                self_att_merge_layer=self_att_merge_layer,
                adv_bias_step=adv_bias_step if i in enable_adv_bias_layers else 0.0,
                shuffle_merge=shuffle_merge, shuffle_merge_type=shuffle_merge_type,
                adv_gradient_boost=adv_gradient_boost,
                adv_gradient_boost_step=adv_gradient_boost_step,
                adv_gradient_boost_func=adv_gradient_boost_func,
                adv_gradient_boost_no_ce=adv_gradient_boost_no_ce,
                gradient_boost_scale=gradient_boost_scale,
                boost_adv_method_list=boost_adv_method_list)
             for i in range(num_layers)])

        logger.info("Boost type: {}.".format(boost_type))
        logger.info("Boost num: {}.".format(num_boost))
        logger.info("Boost main: {}".format(boost_main))
        logger.info("Boost drop rate: {}".format(boost_drop_rate))
        logger.info("Learnable weights: {}".format(not disable_learnable_weights))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.take_dict_input = True
        self.adv_gradient_boost = adv_gradient_boost

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
            opt.num_boost,
            opt.boost_type,
            opt.boost_main,
            opt.boost_drop_rate,
            opt.boost_dropout_diff,
            opt.boost_with_ffn,
            opt.boost_str,
            opt.boost_gating,
            opt.mask_pos_type,
            opt.disable_learnable_weights,
            opt.self_att_merge_layer,
            opt.adv_bias_step,
            opt.adv_bias_layer,
            opt.shuffle_merge,
            opt.shuffle_merge_type,
            opt.adv_gradient_boost,
            opt.adv_gradient_boost_step,
            opt.adv_gradient_boost_func,
            opt.adv_gradient_boost_no_ce,
            opt.adv_gradient_boost_scale,
            opt.boost_adv_method_list)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        kwargs = dict()
        if isinstance(src, dict):
            src_dict = src
            src = src_dict['src']
            # check if use pos
            if 'src_pos' in src_dict:
                kwargs['src_pos'] = src_dict['src_pos'].to(src.device)
                src_pos_lengths = src_dict['src_pos_lengths']
                assert torch.equal(src_pos_lengths, lengths.to(src_pos_lengths.device))
            if 'enc_layer_grads' in src_dict:
                kwargs['enc_layer_grads'] = None
        
        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        extra_loss = torch.tensor(0.0).to(src.device)

        encoder_layer_outputs = []
        for i, layer in enumerate(self.transformer):
            # indicating the second pass.
            if 'enc_layer_grads' in kwargs:
                kwargs['enc_layer_grads'] = src_dict['enc_layer_grads'][i]
            out_dict = layer(out, mask, **kwargs)
            out = out_dict['out']
            extra_loss = out_dict['extra_loss'] + extra_loss
            if self.adv_gradient_boost and 'outs' in out_dict:
                encoder_layer_outputs.append(out_dict['outs'])

        out = self.layer_norm(out)
        ret = {
            'out': out.transpose(0, 1).contiguous(),
            'extra_loss': extra_loss,
            'encoder_layer_outputs': encoder_layer_outputs
        }

        return emb, ret, lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
