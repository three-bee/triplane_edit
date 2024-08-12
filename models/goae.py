import torch
import torch.nn as nn
from torchvision import transforms
from enum import Enum
from models.attention import CrossAttention
from models.afa import AFA
from models.swin_transformer import build_model
from training.triplane import TriPlaneGenerator
import dnnlib
import legacy
from torch_utils import misc

"""
https://github.com/jiangyzy/GOAE
"""

def get_mlp_layer(in_dim, out_dim, mlp_layer=2):
    module_list = nn.ModuleList()
    for j in range(mlp_layer-1):
        module_list.append(nn.Linear(in_dim, in_dim))
        module_list.append(nn.LeakyReLU())
    module_list.append(nn.Linear(in_dim, out_dim)) 
    return nn.Sequential(*module_list)

class GOAEncoder(nn.Module):
    def __init__(self, swin_config, mlp_layer=2, ws_dim = 14, stage_list=[20000, 40000, 60000], type_e='FFHQ'):
        super(GOAEncoder, self).__init__()
        self.style_count = ws_dim       
        self.stage_list = stage_list
        self.stage_dict =   {'base':0, 'coarse':1, 'mid':2, 'fine':3} 
        self.stage = 3
        self.type_e = type_e

## -------------------------------------------------- base w0 swin transformer -------------------------------------------
        if self.type_e == 'FFHQ':
            self.swin_model = build_model(swin_config)#self.swin_model = build_model(swin_config)
        elif self.type_e == 'AFHQ':
            self.swin_base = build_model(swin_config)
        self.mapper_base_spatial = get_mlp_layer(64, 1, mlp_layer)
        self.mapper_base_channel = get_mlp_layer(1024, 512, mlp_layer)

        self.maxpool_base = nn.AdaptiveMaxPool1d(1)

## -------------------------------------------------- w Query mapper coarse mid fine  1024*64 -> (4-1)*512 3*512 7*512 -------------------------------------------
        self.maxpool_query = nn.AdaptiveMaxPool1d(1)

        self.mapper_query_spatial_coarse = get_mlp_layer(64, 3, mlp_layer)
        self.mapper_query_channel_coarse = get_mlp_layer(1024, 512, mlp_layer)
        self.mapper_query_spatial_mid = get_mlp_layer(64, 3, mlp_layer)
        self.mapper_query_channel_mid = get_mlp_layer(1024, 512, mlp_layer)
        self.mapper_query_spatial_fine = get_mlp_layer(64, 7, mlp_layer)
        self.mapper_query_channel_fine = get_mlp_layer(1024, 512, mlp_layer)

## -------------------------------------------------- w KQ coarse mid fine mapper to 512 -------------------------
        self.mapper_coarse_channel = nn.Sequential(nn.Linear(512,512), nn.LeakyReLU())      
        self.mapper_mid_channel = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU())         
        self.mapper_fine_channel = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(), nn.Linear(256,512), nn.LeakyReLU())   

        self.mapper_coarse_to_mid_spatial = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU(), nn.Linear(512,1024), nn.LeakyReLU())
        self.mapper_mid_to_fine_spatial = nn.Sequential(nn.Linear(1024,2048), nn.LeakyReLU(), nn.Linear(2048,4096), nn.LeakyReLU())

## -------------------------------------------------- w KQ coarse mid fine Cross Attention -------------------------
        self.cross_att_coarse = CrossAttention(512, 4, 1024, batch_first=True)
        self.cross_att_mid = CrossAttention(512, 4, 1024, batch_first=True)
        self.cross_att_fine = CrossAttention(512, 4, 1024, batch_first=True)

    def forward(self, x):
        B = x.shape[0]
        if self.type_e == 'AFHQ':
            x_base, x_query, x_coarse, x_mid, x_fine= self.swin_base(x) #self.swin_model(x)
        elif self.type_e == 'FFHQ':
            x_base, x_query, x_coarse, x_mid, x_fine= self.swin_model(x)

## ----------------------  base 
        ws_base_max = self.maxpool_base(x_base).transpose(1, 2)     
        ws_base_linear = self.mapper_base_spatial(x_base)     
        ws_base = self.mapper_base_channel(ws_base_linear.transpose(1, 2)  + ws_base_max )   

        ws_base = ws_base.repeat(1,14,1)     

        if self.stage == self.stage_dict['base']:
            ws = ws_base
            return ws, ws_base
## ------------------------ coarse mid fine ---  query    
        ws_query_max = self.maxpool_query(x_query).transpose(1, 2)  

        if self.stage >= self.stage_dict['coarse']:
            ws_query_linear_coarse = self.mapper_query_spatial_coarse(x_query)    
            ws_query_coarse = self.mapper_query_channel_coarse(ws_query_linear_coarse.transpose(1, 2)  + ws_query_max ) 

            if self.stage >= self.stage_dict['mid']:
                ws_query_linear_mid = self.mapper_query_spatial_mid(x_query)      
                ws_query_mid = self.mapper_query_channel_mid(ws_query_linear_mid.transpose(1, 2)  + ws_query_max )   

                if self.stage >= self.stage_dict['fine']:
                    ws_query_linear_fine = self.mapper_query_spatial_fine(x_query)  
                    ws_query_fine = self.mapper_query_channel_fine(ws_query_linear_fine.transpose(1, 2)  + ws_query_max )  

## -------------------------  carse, mid, fine -----  key-value 
        if self.stage >= self.stage_dict['coarse']:
            kv_coarse = self.mapper_coarse_channel(x_coarse)     

            if self.stage >= self.stage_dict['mid']:
                kv_mid = self.mapper_mid_channel(x_mid) + self.mapper_coarse_to_mid_spatial(kv_coarse.transpose(1, 2)).transpose(1, 2)    

                if self.stage >= self.stage_dict['fine']:
                    kv_fine = self.mapper_fine_channel(x_fine) + self.mapper_mid_to_fine_spatial(kv_mid.transpose(1, 2)).transpose(1, 2) 
        
## ------------------------- carse, mid, fine -----  Cross attention
        if self.stage >= self.stage_dict['coarse']:
            ws_coarse = self.cross_att_coarse(ws_query_coarse, kv_coarse )
            zero_1 = torch.zeros(B,1,512).to(ws_base.device)
            zero_2 = torch.zeros(B,10,512).to(ws_base.device)
            ws_delta = torch.cat([zero_1, ws_coarse, zero_2], dim=1)

            if self.stage >= self.stage_dict['mid']:
                ws_mid = self.cross_att_mid(ws_query_mid, kv_mid)
                zero_1 = torch.zeros(B,1,512).to(ws_base.device)
                zero_2 = torch.zeros(B,7,512).to(ws_base.device)
                ws_delta = torch.cat([zero_1, ws_coarse, ws_mid, zero_2], dim=1)


                if self.stage >= self.stage_dict['fine']:
                    ws_fine = self.cross_att_fine(ws_query_fine, kv_fine)
        
                    zero = torch.zeros(B,1,512).to(ws_base.device)

                    ws_delta = torch.cat([zero, ws_coarse, ws_mid, ws_fine], dim=1)

        ws = ws_base + ws_delta
        return ws, ws_base