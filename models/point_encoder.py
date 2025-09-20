import torch
import torch.nn as nn
import torch.nn.functional as F

"""
  idea：不用随机初始化的类别编码，选择当前选择点周围的点进行采样然后拼接成tensor作为类别编码，因为
  教师网络其实主要就是为了预测定位框，所以如果只是输入一个随机的类别编码对于预测边界框没有帮助
"""

"""
   预训练任务和该模型的结合 不能单纯的用之前预训练的权重来作为教师模型的训练权重，因为之前的模型没有包含
   point编码模块，所以之前的权重相当于只预训练了detr部分。所以还是要做适配的。即在预训练阶段不仅要让模型
   学会定位，而且还要让模型学会从点坐标预测框的坐标。所以结合之前的任务可以做如下调整：1.构造合成图的方式
   不变，合成图贴上patch的地方即为gt 坐标。2.从贴上patch的框内随机选择一点作为输入点编码器的输入，然后
   损失函数部分，只预测边界框即可。3、对于特征做的一些改变不动，比如数据增强，增加绿色等。4、这样整个模型
   都被预训练了，而且模型不仅去学习定位物体了，而且也学习了这些特征表示，而且也预训练了点编码器。
"""


class PointEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.query_emb = nn.Embedding(100, 256)
        self.label_emb = nn.Linear(2048, 256)

    #Points , Nx2
    #labels , N,
    #object_ids N,
    # def forward(self, batched_points, batched_labels, batched_object_ids, pos_encoder, label_encoder):
    def forward(self, src, val_stage,points_supervision, samples_width_height, pos_encoder, label_encoder, no_label_enc, no_pos_enc, device):
        # batch_size = len(batched_points)
        batch_size = len(points_supervision)
        #position embedding .... by points
        #label embedding ... by labels
        #... object embedding ??  ...
        #feature embedding .. by points && features (interpolation)
        label_embs = []
        points = []
        for ii in range(batch_size):
            points.append(points_supervision[ii]["points"])
        for idx in range(batch_size):
            n_p, c = points[idx].shape
            ps = points[idx].view(1, 1, n_p, c)
            feature = src[idx].unsqueeze(0)
            sampling_grids = 2 * ps - 1
            # N_*M_, D_, Lq_, P_
            # 采样出给定点在特征图上2048维的特征 然后进行编码成256
            sampling_value = F.grid_sample(feature, sampling_grids,
                                           mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value = sampling_value.view(sampling_value.shape[3], sampling_value.shape[1])
            label_emb = self.label_emb(sampling_value)
            label_embs.append(label_emb)
        embeddings = []
        for idx in range(batch_size):
            points = points_supervision[idx]['points']
            width, height = samples_width_height[idx]
            l = points[:, 0] - 0.5 * width + 0.5 * width
            t = points[:, 1] - 0.5 * height + 0.5 * height
            r = width - l
            b = height - t
            relative_sites = torch.stack([l / r, t / b], dim=1)
            # if isinstance([relative_sites_nested], (list, torch.Tensor)):
            #     relative_sites_nested = nested_tensor_from_tensor_list([relative_sites_nested])

            relative_embedding = pos_encoder.calc_emb(relative_sites)


            position_embedding = pos_encoder.calc_emb(points_supervision[idx]['points'])
            if no_label_enc:
                label_embedding = torch.zeros((position_embedding.size())).to(device)
                # N = len(points_supervision[idx]['points'])
                # label_embedding = pos_encoder.calc_emb(rand_pos)
            else:
                label_embedding = label_encoder.calc_emb(points_supervision[idx]['labels'])
            if no_pos_enc:
                position_embedding = torch.zeros((position_embedding.size())).to(device) #..

            query_embedding = position_embedding + label_embedding
            #query_embedding = query_embedding * label_embs[idx]
            if no_label_enc and no_pos_enc:
                # print('position_embedding .. size ', query_embedding.size())
                N = len(position_embedding)
                query_embedding = self.query_emb.weight[:N]
                # print('query embedding .. size ', query_embedding.size())

            embeddings.append(query_embedding)

            # print('label embedding', label_embedding)
            # print('label_embedding', label_embedding.mean(), label_embedding.var(), label_embedding.size())
            # print('position embedding .. ', position_embedding.mean(), position_embedding.var(), position_embedding.size())
            # embeddings.append(label_embedding)
        return embeddings



def build_point_encoder():
    return PointEncoder()
