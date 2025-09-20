# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import pdb
from .attention import MultiheadAttention

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape   # [2,256,27,24]
        # pdb.set_trace()
        src = src.flatten(2).permute(2, 0, 1) #hwx2x256 [648,2,256]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #hwx2x256  [648,2,256]
        # pdb.set_trace()
        # query_embed  4x256,  15x256

        positive_num = [query_embed[idx].size(0) for idx in range(bs)] #4, 15
        # print('batch num .. ' , positive_num)


        padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)] #15 - [4, 15]
        # print('padding num .. ', padding_num)




        for idx in range(bs):
            if padding_num[idx] == 0:
                continue
            query_embed[idx] = torch.cat([query_embed[idx], torch.zeros((padding_num[idx], c), device=src.device)], dim=0)

        query_embed = torch.stack(query_embed, dim=1)   # [3,2,256]
        # pdb.set_trace()
        # print('query_embed', query_embed.size())

        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #100x2x256
        mask = mask.flatten(1) #2xhw

        # print('src, pos_embed, query_embed, mask... ', src.size(), pos_embed.size(), query_embed.size(), mask.size())

        # tgt = torch.zeros_like(query_embed)  初始时将tgt赋予point的编码，希望使用点的信息来做一个好的初始化
        tgt = query_embed
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # [648,2,256]
        # pdb.set_trace()
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # pdb.set_trace()
        ans = []

        for idx in range(bs):
            ans.append(hs[:, :positive_num[idx], idx, :])
            # print('ans ... idx', ans[idx].size())
        return ans


# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path C:\Users\jack\Desktop\PointDETR-modify\PointDETR-main\MSCoco --partial_training_data --output_dir ./ckpt-ps/point-detr-9x --epochs 108 --lr_drop 72 --data_augment --position_embedding sine --warm_up --multi_step_lr

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        d_model = 256
        self.query_scale = MLP(d_model, d_model, d_model, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        # pdb.set_trace()
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
                # pdb.set_trace()
            else:
                # FFN映射tgt
                pos_transformation = self.query_scale(output)
                # pdb.set_trace()
            query_sine_embed = query_pos * pos_transformation   # [3,2,256]
            # pdb.set_trace()
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # print('transformer .. forward_post')
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):

        # print('transformer .. forward_pre')
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.nhead = nhead

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.multihead_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)   # 因为query和key均发生了改变，进行了拼接，所以其维度变为d_model*2
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed=None,  # 根据conditional-detr论文中的获取方式获取 point的编码和tgt经过投影后进行相乘
                     is_first=False  # 是否是第一层
                     ):
        # self-attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # print("1",tgt.shape)
        # pdb.set_trace()
        sq_content = self.sa_qcontent_proj(tgt)  #  [3,2,256] target is the input of the first decoder layer. zero by default.
        sq_pos = self.sa_qpos_proj(query_pos)  # [3.2.256]
        # pdb.set_trace()
        sk_content = self.sa_kcontent_proj(tgt) # [3,2,256]
        # print("11",sk_content.shape)
        sk_pos = self.sa_kpos_proj(query_pos)   # [3,2,256]
        sv = self.sa_v_proj(tgt)                # [3,2,256]
        # pdb.set_trace()
        num_queries, bs, n_model = sq_content.shape
        hw, _, _ = sk_content.shape
        # pdb.set_trace()
        sq = sq_content + sq_pos  # [3,2,256]
        sk = sk_content + sk_pos  # [3,2,256]
        # pdb.set_trace()
        tgt2 = self.self_attn(sq, sk, value=sv, attn_mask=tgt_mask, # [3,2,256]
                              key_padding_mask=tgt_key_padding_mask)[0]
        # pdb.set_trace()
        # print("2", tgt2.shape)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)   # [3,2,256]
        # print("3", tgt.shape)
        # pdb.set_trace()
        # cross-attention 参考conditional-detr改造
        cq_content = self.ca_qcontent_proj(tgt)   # query 的内容部分也就是self-attention的输出 [3,2,256]
        # pdb.set_trace()
        # print("4", cq_content.shape)
        ck_content = self.ca_kcontent_proj(memory)  # encoder的输出    [648,2,256]
        # print("5", ck_content.shape)
        cv = self.ca_v_proj(memory)          # encoder的输出           [648,2,256]
        # pdb.set_trace()
        num_queries, bs, n_model = cq_content.shape
        # pdb.set_trace()
        hw, _, _ = ck_content.shape
        ck_pos = self.ca_kpos_proj(pos)         # cross-attention的key的位置编码  [648,2,256]
        # print("6", ck_pos.shape)
        # pdb.set_trace()
        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            cq_pos = self.ca_qpos_proj(query_pos)   # [3,2,256]
            cq_combine = cq_content + cq_pos    # [3,2,256]
            ck_combine = ck_content + ck_pos
        else:
            cq_combine = cq_content
            ck_combine = ck_content
        # pdb.set_trace()
        # print("7", cq_combine.shape)
        cq_combine = cq_combine.view(num_queries, bs, self.nhead, n_model // self.nhead)  # 考虑到多头head   [3,2,8,32]
        # print("8", cq_combine.shape)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed) #[3,2,256]
        # pdb.set_trace()
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)  # 根据conditional-detr获取到的额外编码  [3,2,8,32]
        # pdb.set_trace()
        # print("9", query_sine_embed.shape)
        cq_combine = torch.cat([cq_combine, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2) # 拼接起来 而不是相加   [3,2,512]
        # print("10", cq_combine.shape)
        ck_combine = ck_combine.view(hw, bs, self.nhead, n_model // self.nhead) # [648,2,8,32]
        # print("11", ck_combine.shape)
        ck_pos = ck_pos.view(hw, bs, self.nhead, n_model // self.nhead) # key的位置编码  [648,2,8,32]
        # print("12", ck_pos.shape)
        ck_combine = torch.cat([ck_combine, ck_pos], dim=3).view(hw, bs, n_model * 2) # 也是进行拼接 [2,648,512]
        # cv = cv.view(bs, hw, n_model)
        # print("13", ck_combine.shape)
        # print(cv.shape)
        tgt2 = self.multihead_attn(query=cq_combine,
                               key=ck_combine,
                               value=cv, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]   #[3,2,256]
        # pdb.set_trace()
        # print("14", tgt2.shape)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    query_sine_embed = None,  # 根据conditional-detr论文中的获取方式获取 point的编码和tgt经过投影后进行相乘
                    is_first = False  # 是否是第一层
                    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,  # 根据conditional-detr论文中的获取方式获取 point的编码和tgt经过投影后进行相乘
                is_first=False
                ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,query_sine_embed,is_first)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path C:\Users\jack\Desktop\PointDETR-modify\PointDETR-main\MSCoco\  --partial_training_data --output_dir ./ckpt-ps/point-detr-9x --epochs 108 --lr_drop 72 --data_augment --position_embedding sine --warm_up --multi_step_lr
