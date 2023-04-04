# Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

from .model_utils import *
import pdb


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, res_expansion):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = ConvBNReLU1D(out_dim * expand, out_dim, bias=False, activation='relu')
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(ConvBNReLURes2D(out_dim, res_expansion=res_expansion, bias=True, activation='relu'))
        self.linear2 = nn.Sequential(*self.linear2)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalize x (features) and xyz (coordinates)
        # mean_x = lc_x.unsqueeze(dim=-2)
        # std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        # knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G*K)).reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        # self.out_transform = nn.Sequential(
        #         nn.BatchNorm1d(out_dim),
        #         nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        # lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding 
class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer
class ConvBNReLURes2D(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes2D, self).__init__()

        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * res_expansion),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels * res_expansion), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)
    

# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        # position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        #position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w

    
# Parametric Encoder
class EncP(nn.Module):  
    def __init__(self, in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, res_expansion):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = ConvBNReLU1D(in_channels, self.embed_dim, bias=False, activation='relu')

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, LGA_block[i], dim_expansion[i], res_expansion))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


# Parametric Network for ModelNet40
class Point_PN_mn40(nn.Module):
    def __init__(self, in_channels=3, class_num=40, input_points=1024, num_stages=4, embed_dim=36, k_neighbors=40, beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,1], res_expansion=0.5):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncP(in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, res_expansion)
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            get_activation('relu'),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            get_activation('relu'),
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )

    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)

        # Classifier
        x = self.classifier(x)
        return x
    

# Parametric Network for ScanObjectNN
class Point_PN_scan(nn.Module):
    def __init__(self, in_channels=4, class_num=15, input_points=1024, num_stages=4, embed_dim=36, k_neighbors=40, beta=100, alpha=1000, LGA_block=[2,1,1,1], dim_expansion=[2,2,2,1], res_expansion=0.5):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncP(in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, res_expansion)
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            get_activation('relu'),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            get_activation('relu'),
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )

    def forward(self, x, xyz):
        # xyz: point coordinates
        # x: point features
        # xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)

        # Classifier
        x = self.classifier(x)
        return x