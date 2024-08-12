import os
import torch
import torchvision
import dnnlib
import argparse
import dnnlib
import random
import legacy
import json
import copy

import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch_utils import misc
from configs.swin_config import get_config
from torch.backends import cudnn
from training.triplane import TriPlaneGenerator
from models.goae import GOAEncoder, AFA
from camera_utils import LookAtPoseSampler

from face_parsing_pytorch.model import BiSeNet

from training.ranger import Ranger
from criteria.metrics import Metrics

class TriplaneEditingPipeline:
    def __init__(self, opts: argparse.Namespace, device='cuda', outdir=None):
        super(TriplaneEditingPipeline, self).__init__()
        self.set_seeds(1453)

        self.opts = opts
        self.device = device
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        
        ## Networks
        self.decoder, self.w_avg = self.set_eg3d_generator(G_ckpt_path=opts.G_ckpt_path, device=device)
        if opts.Efinetuned_ckpt_path is not None:
            self.encoder_frozen = self.set_encoder(E_ckpt_path=opts.Efinetuned_ckpt_path, swin_config=get_config(opts), device=device, trainable=False)
            print(f'Loaded finetuned encoder {opts.Efinetuned_ckpt_path}!')
        else:
            self.encoder_frozen = self.set_encoder(E_ckpt_path=opts.E_ckpt_path, swin_config=get_config(opts), device=device, trainable=False)
        self.encoder_second_stage_frozen = self.set_second_stage_encoder(E_ckpt_path=opts.E2_ckpt_path, device=device, trainable=False)
        self.masking_net = self.set_mask_net(opts.mask_gen_ckpt, device)
        self.attribute_chs = {'mouth':[11,12,13],
                              'glasses':[6],
                              'hair':[17],
                              'brows':[2,3],
                              'eyes':[4,5],
                              'nose':[10]}
        self.w_glasses = torch.from_numpy(np.load('./checkpoints/glass.npy').reshape(14,512)).to(self.device)

        ## Criteria for runtime optim
        self.metrics = Metrics(ir_se_50_path=opts.ir_se_50_path, device=device)

        ## Rendering stuff
        self.to_256 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.to_512 = torch.nn.Upsample(size=(512,512), mode='bilinear', align_corners=True)
        self.cam_pivot = torch.tensor([0, 0, 0.2], device=self.device)
        self.intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1], device=self.device)
        self.c_front = torch.tensor([1,0,0,0, 
                        0,-1,0,0,
                        0,0,-1,2.7, 
                        0,0,0,1, 
                        4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(self.device).reshape(1,-1)
        autograd_yaw_pitch_list = [(-60,5),(-45,5),(-30,0),(-15,10),(0,15),(0,0),(0,-15),(15,10),(30,0),(45,5),(60,5)]
        self.novel_pose_list = [self.get_pose(intrinsics=self.intrinsics, cam_pivot=self.cam_pivot, yaw=y*np.pi/180, pitch=p*np.pi/180) for y,p in autograd_yaw_pitch_list]
        visualize_yaw_pitch_list = [(-30,10),(-15,10),(0,10),(15,10),(30,10)]
        self.novel_pose_list_visualize = [self.get_pose(intrinsics=self.intrinsics, cam_pivot=self.cam_pivot, yaw=y*np.pi/180, pitch=p*np.pi/180) for y,p in visualize_yaw_pitch_list]
    
    @staticmethod
    def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7, device='cuda'):
        if yaw is None:
            yaw = np.random.uniform(-yaw_range, yaw_range)
        if pitch is None:
            pitch = np.random.uniform(-pitch_range, pitch_range)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
        return c

    @staticmethod
    def set_seeds(seed):
        cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def set_mask_net(mask_gen_ckpt, device):
        mask_net = BiSeNet(n_classes=19).to(device).requires_grad_(False)
        mask_ckpt = torch.load(mask_gen_ckpt, map_location=torch.device(device))
        mask_net.load_state_dict(mask_ckpt)
        print(f'Loaded {mask_gen_ckpt}!')
        del mask_gen_ckpt
        mask_net = mask_net.eval()
        return mask_net
    
    @staticmethod
    def create_2d_roi_mask(img_rec, roi_channel, masking_net):    
        parsing = masking_net(img_rec)[0]
        parsing = parsing.squeeze(0).argmax(0)
        roi_mask = torch.zeros_like(parsing).float()
        for ch in roi_channel:
            roi_mask[parsing==ch] = 1.0
        rec_roi_removed = img_rec*roi_mask
        return roi_mask

    @staticmethod
    def set_eg3d_generator(G_ckpt_path, device):
        with dnnlib.util.open_url(G_ckpt_path) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        print(f'Loaded {G_ckpt_path}!')
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        del G
        return G_new, G_new.backbone.mapping.w_avg[None, None, :].to(device)
    
    @staticmethod
    def set_encoder(E_ckpt_path, swin_config, device, trainable=False):
        if trainable:
            E = GOAEncoder(swin_config, mlp_layer=2, stage_list=[10000, 20000, 30000]).to(device)
        else:
            E = GOAEncoder(swin_config, mlp_layer=2, stage_list=[10000, 20000, 30000], type_e='FFHQ').eval().requires_grad_(False).to(device)
        if E_ckpt_path:
            E_ckpt = torch.load(E_ckpt_path, map_location=torch.device(device))
            E.load_state_dict(E_ckpt, strict=True)
            print(f'Loaded {E_ckpt_path}!')
        return E

    @staticmethod
    def set_second_stage_encoder(E_ckpt_path, device, trainable=False):
        if trainable:
            E2 = AFA().to(device)
        else:
            E2 = AFA().eval().requires_grad_(False).to(device)
        if E_ckpt_path:
            AFA_ckpt = torch.load(E_ckpt_path, map_location=torch.device(device))
            E2.load_state_dict(AFA_ckpt, strict=True)
        print(f'Loaded {E_ckpt_path}!')
        return E2
    
    def autograd_create_roi_mask(self, novel_pose_list, w_input, tp_input, decoder, roi_channel, masking_net):
        """
        Renders the decoded triplane from different views and accumulates the triplane gradients.
        If tp_input is given, w_input is ignored for the triplane generation part. For the super-resolution stage, w_input is always utilized.
        """
        grad_list = []
        for idx, pose in enumerate(novel_pose_list):
            decoder.requires_grad_(True)
            if w_input is not None:
                rec_list = decoder.synthesis(w_input, pose, noise_mode='const', return_triplanes=True)
            elif tp_input is not None: # If triplanes are given, overwrite rec_list
                tp_input.requires_grad_(True)
                rec_list = decoder.synthesis(w_input, pose, noise_mode='const', replace_triplane=True, tp=tp_input, return_triplanes=True)
            
            rec = rec_list['image']
            rec_triplane = rec_list['planes']
            roi_mask = self.create_2d_roi_mask(img_rec=rec, roi_channel=roi_channel, masking_net=masking_net)
            grad = torch.autograd.grad(outputs=rec, inputs=rec_triplane, grad_outputs=roi_mask.unsqueeze(0).repeat(1,3,1,1))[0]            
            grad_list.append(grad)

            decoder.requires_grad_(False)
            if tp_input is not None:
                tp_input.requires_grad_(False)
        
        grad_mask = torch.mean(torch.stack(grad_list, dim=0),dim=0)
        return grad_mask
    
    @staticmethod
    def postprocess_tp_mask(tp_grad, mean_seperation=True, mean_range=(0.9, 1.1),
                            smooth=True, blur_func=torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0)), blur_iter=3,
                            binarize=True):
        """
        Apply postprocessing steps to the output of autograd_create_roi_mask().
        Specifically, perform channelwise minmax normalization, mean seperation, and binarization to triplane gradients.
        """
        t = tp_grad.view(-1,256,256)
        for i in range(t.shape[0]): # channel-wise minmax norm   
            min_ele = torch.min(t[i])
            t[i] -= min_ele
            t[i] /= torch.max(t[i])

        if mean_seperation:
            for i in range(t.shape[0]):
                mu = t[i].mean()
                t[i] = (t[i]<mean_range[1]*mu) * (t[i]>mean_range[0]*mu) # take around mean region
                if smooth:
                    for _ in range(blur_iter):
                        t[i] = blur_func(t[i].unsqueeze(0)).squeeze(0)
        t = 1-t # get the inverse mask since we are interested in outside of the mean

        if binarize:
            mean_grad = torch.mean(t[0:32],dim=0).unsqueeze(0).repeat(96,1,1)
            b_mask = torch.zeros_like(mean_grad)
            b_mask[mean_grad>mean_grad.mean()] = 1.0
            if smooth:
                b_mask = blur_func(b_mask.unsqueeze(0)).squeeze(0)
            t = b_mask

        t = t.view(1,3,32,256,256)
        return t

    # https://github.com/jiangyzy/GOAE/blob/7886952f11007a77d2e4daec3788940dea0d8cc8/goae/training/goae.py#L264  
    def mix_fusion(self, bsz, rec_img_dict, rec_img_dict_w):
        blur_E = torchvision.transforms.GaussianBlur(9, sigma=(1,2))  

        with torch.no_grad():
            ray_origins = rec_img_dict['ray_origins']
            ray_directions = rec_img_dict['ray_directions']
            depth = rec_img_dict['image_depth'].flatten(start_dim=1).reshape(bsz, -1, 1).detach()
            xyz = ray_origins + depth * ray_directions

            x = (xyz[:,:,0] + 0.5)*256
            y = (xyz[:,:,1] + 0.5)*256
            x = torch.clamp(x, min=0, max=255)
            y = torch.clamp(y, min=0, max=255)
            x = x.long()
            y = y.long()

            mask = torch.zeros((bsz,1,256,256))
            for j in range(bsz):
                mask[j, :, y[j], x[j]] = 1
                mask[j] = blur_E(mask[j])
                mask[j] = torch.where(mask[j] > 0.35, 1, 0).float()
                mask[j] = blur_E(mask[j])
        mask = mask.to(self.device)

        f_triplane = rec_img_dict['triplane']
        w_triplane = rec_img_dict_w['triplane']  
        mix_triplane = torch.zeros_like(f_triplane)   

        mix_triplane[:,0,:,:,:] = mask * f_triplane[:,0,:,:,:] + (1-mask) * w_triplane[:,0,:,:,:]
        mix_triplane[:,1:,:,:,:] = f_triplane[:,1:,:,:,:]       

        return mix_triplane, mask

    # https://github.com/jiangyzy/GOAE/blob/7886952f11007a77d2e4daec3788940dea0d8cc8/goae/training/goae.py#L230
    def forward_encoder_allstages(self, x, x_512, c, E1, E2):
        B = x.shape[0]

        rec_ws, _ = E1(x)
        rec_ws = rec_ws + self.w_avg.repeat(B, 1, 1)
        triplane, triplane_x = self.decoder.synthesis(ws=rec_ws, c=c, train_forward_1=True, noise_mode='const')
        rec_img_dict_w = self.decoder.synthesis(ws=rec_ws, c=c, triplane=triplane, triplane_x=triplane_x, train_forward_2=True, noise_mode='const')
        feature_map_adain, gamma, beta = E2(x_512, rec_img_dict_w['image'], triplane_x)
        rec_img_dict = self.decoder.synthesis(ws=rec_ws, c=c, triplane=triplane, triplane_x=feature_map_adain, train_forward_2=True, noise_mode='const')  
        mix_triplane, mask = self.mix_fusion(B, rec_img_dict, rec_img_dict_w)     

        return rec_ws, mix_triplane

    def synthesise_from_w(self, input_w, pose, return_triplanes=None, add_w_avg=False):
        if add_w_avg:
            input_w = input_w + self.w_avg
        rec = self.decoder.synthesis(input_w, pose, noise_mode='const', return_triplanes=return_triplanes)
        return rec
    
    def synthesise_from_tp(self, input_w, input_tp, pose, return_triplanes=None, add_w_avg=False):
        if add_w_avg:
            input_w = input_w + self.w_avg
        rec = self.decoder.synthesis(input_w, pose, noise_mode='const', replace_triplane=True, tp=input_tp, return_triplanes=return_triplanes)
        return rec
    
    def create_w_tp_from_img(self, img, img_512, c, E1, E2):
        w, t = self.forward_encoder_allstages(x=img, x_512=img_512, c=c, E1=E1, E2=E2)
        return w, t
    
    def create_dilated_eroded_tp_masks(self, tp_mask, blur_k_size=9, morph_k_size=11, std_devs=(2.0,2.0)):
        blur = torchvision.transforms.GaussianBlur(blur_k_size, sigma=std_devs)
        dilated_tp_mask = blur(F.max_pool2d(tp_mask.view(1,96,256,256), kernel_size=morph_k_size, stride=1, padding=(morph_k_size - 1) // 2)).view(1,3,32,256,256)
        eroded_tp_mask = blur((1 - F.max_pool2d((1 - tp_mask.view(1,96,256,256)), kernel_size=morph_k_size, stride=1, padding=(morph_k_size - 1) // 2))).view(1,3,32,256,256)
        return eroded_tp_mask, dilated_tp_mask

    def forward_manual_fusion(self,
                              w_src, t_src, w_dst, t_dst,
                              decoder, masking_net, 
                              src_roi_ch, dst_roi_ch, 
                              use_dst_mask_in_src=False,
                              smooth_tp=True,
                              invert_src_mask=False, invert_dst_mask=False, 
                              w_guidance_dir=None, w_guidance_factor=1.25):
        """
        Performs manual fusion (t_mask_dst*t_dst + t_mask_src*t_src) of source and destination triplanes using specified masks.
        """                              
        src_tp_roi = self.autograd_create_roi_mask(novel_pose_list=self.novel_pose_list,
                                                   w_input=w_src, tp_input=t_src, decoder=decoder,
                                                   roi_channel=src_roi_ch, masking_net=masking_net)
        src_tp_roi = self.postprocess_tp_mask(tp_grad=src_tp_roi, smooth=smooth_tp, mean_seperation=True, binarize=True)
        
        if use_dst_mask_in_src:
            dst_tp_roi = self.autograd_create_roi_mask(novel_pose_list=self.novel_pose_list,
                                                       w_input=w_dst, tp_input=t_dst, decoder=decoder,
                                                       roi_channel=dst_roi_ch, masking_net=masking_net)
            dst_tp_roi = self.postprocess_tp_mask(tp_grad=dst_tp_roi, mean_seperation=True, binarize=True)

        if invert_src_mask:
            src_tp_roi = 1-src_tp_roi
        if invert_dst_mask:
            dst_tp_roi = 1-dst_tp_roi

        t_mask_src = src_tp_roi*dst_tp_roi if use_dst_mask_in_src else src_tp_roi
        t_mask_dst = 1-t_mask_src

        if w_guidance_dir is not None:
            t_edited = decoder.synthesis(w_src-w_guidance_dir, self.c_front, noise_mode="const", return_triplanes=True)['planes']
            t_mask_dst = 1
            t_src = w_guidance_factor*(t_src - t_edited)            

        man_fused_tp = t_mask_dst*t_dst + t_mask_src*t_src
        return man_fused_tp, t_mask_src, t_mask_dst

    def forward_implicit_fusion(self, encoder, w_src, man_fused_tp=None, rec_fused_front=None):
        """
        Renders man_fused_tp from canonical view, and passes the image through encoder + triplane decoder to implicitly fuse man_fused_tp.
        If rec_fused_front is explicitly given, then the initial canonical view rendering is bypassed.
        """
        if rec_fused_front is None and man_fused_tp is not None:
            rec_fused_front = self.synthesise_from_tp(input_w=w_src, input_tp=man_fused_tp, pose=self.c_front, return_triplanes=False)['image']
        w_fused_front, _ = encoder(self.to_256(rec_fused_front))
        imp_fused_tp = self.synthesise_from_w(input_w=w_fused_front, pose=self.c_front, return_triplanes=True, add_w_avg=True)['planes']
        return w_fused_front, imp_fused_tp

    def reference_tp_edit(self, edit_label, w_src, t_src, w_dst, t_dst, guidance_factor=1.5):
        ## Manual fusion
        if edit_label == 'hair':
            man_fused_tp, t_mask_src, t_mask_dst = self.forward_manual_fusion(w_src, t_src, w_dst, t_dst, self.decoder, self.masking_net,
                                                                              src_roi_ch=self.attribute_chs['hair'], dst_roi_ch=self.attribute_chs['hair'],
                                                                              smooth_tp=True, invert_src_mask=True, invert_dst_mask=True, use_dst_mask_in_src=True)
            eroded_src_tp_mask, dilated_src_tp_mask = self.create_dilated_eroded_tp_masks(t_mask_src, blur_k_size=21, morph_k_size=11, std_devs=(11.0,11.0))
            eroded_dst_tp_mask, dilated_dst_tp_mask = self.create_dilated_eroded_tp_masks(t_mask_dst, blur_k_size=21, morph_k_size=11, std_devs=(11.0,11.0))
            
            M_src = eroded_src_tp_mask
            M_imp_mid = 1-(eroded_src_tp_mask+eroded_dst_tp_mask)
            M_dst = eroded_dst_tp_mask
            M_man = 0
            w_sr = w_src
        elif edit_label == 'glasses':
            man_fused_tp, t_mask_src, t_mask_dst = self.forward_manual_fusion(w_src, t_src, w_dst, t_dst, self.decoder, self.masking_net,
                                                                              src_roi_ch=self.attribute_chs['glasses'], dst_roi_ch=None,
                                                                              smooth_tp=False, use_dst_mask_in_src=False, invert_src_mask=False, invert_dst_mask=False,
                                                                              w_guidance_dir=self.w_glasses, w_guidance_factor=guidance_factor)
            eroded_src_tp_mask, dilated_src_tp_mask = self.create_dilated_eroded_tp_masks(t_mask_src, blur_k_size=31, morph_k_size=3, std_devs=(9.0,9.0)) # NOTE reported
            
            M_src = 0
            M_imp_mid = (dilated_src_tp_mask-eroded_src_tp_mask)
            M_dst = 1-dilated_src_tp_mask
            M_man = eroded_src_tp_mask
            w_sr = w_dst
        elif edit_label == 'eyes':
            man_fused_tp, t_mask_src, t_mask_dst = self.forward_manual_fusion(w_src, t_src, w_dst, t_dst, self.decoder, self.masking_net,
                                                                              src_roi_ch=self.attribute_chs['eyes'], dst_roi_ch=None,
                                                                              smooth_tp=True, use_dst_mask_in_src=False, invert_src_mask=False, invert_dst_mask=False)
            eroded_src_tp_mask, dilated_src_tp_mask = self.create_dilated_eroded_tp_masks(t_mask_src, blur_k_size=9, morph_k_size=11, std_devs=(2.0,2.0))
            
            M_src = eroded_src_tp_mask
            M_imp_mid = dilated_src_tp_mask-eroded_src_tp_mask
            M_dst = 1-dilated_src_tp_mask
            M_man = 0
            w_sr = w_dst
        elif edit_label == 'mouth':
            man_fused_tp, t_mask_src, t_mask_dst = self.forward_manual_fusion(w_src, t_src, w_dst, t_dst, self.decoder, self.masking_net,
                                                                              src_roi_ch=self.attribute_chs['mouth'], dst_roi_ch=None,
                                                                              smooth_tp=True, use_dst_mask_in_src=False, invert_src_mask=False, invert_dst_mask=False)
            eroded_src_tp_mask, dilated_src_tp_mask = self.create_dilated_eroded_tp_masks(t_mask_src, blur_k_size=21, morph_k_size=11, std_devs=(2.0,2.0))
            
            M_src = eroded_src_tp_mask
            M_imp_mid = dilated_src_tp_mask-eroded_src_tp_mask
            M_dst = 1-dilated_src_tp_mask
            M_man = 0
            w_sr = w_dst
        elif edit_label == 'nose':
            man_fused_tp, t_mask_src, t_mask_dst = self.forward_manual_fusion(w_src, t_src, w_dst, t_dst, self.decoder, self.masking_net,
                                                                              src_roi_ch=self.attribute_chs['nose'], dst_roi_ch=self.attribute_chs['eyes'],
                                                                              use_dst_mask_in_src=True, invert_src_mask=False, invert_dst_mask=True)
            eroded_src_tp_mask, dilated_src_tp_mask = self.create_dilated_eroded_tp_masks(t_mask_src, blur_k_size=21, morph_k_size=11, std_devs=(9.0,9.0))
            
            M_src = 0
            M_imp_mid = dilated_src_tp_mask
            M_dst = 1-dilated_src_tp_mask
            M_man = 0
            w_sr = w_dst
        else:
            raise NotImplementedError
        
        ## Implicit fusion
        w_fused_front, imp_fused_tp = self.forward_implicit_fusion(encoder=self.encoder_frozen, w_src=w_src, man_fused_tp=man_fused_tp)

        ## Final fusion
        final_fused_tp = t_src*M_src+imp_fused_tp*M_imp_mid+t_dst*M_dst+man_fused_tp*M_man

        return w_sr, final_fused_tp
    
    def runtime_E_optim(self, img_256, img_512, c, iter_num=50):
        """
        Performs runtime optimization on the F-space encoder. 
        Can be enabled for better reconstruction of src and dst but is not a must.
        """
        E2_trainable = copy.deepcopy(self.encoder_second_stage_frozen).requires_grad_(True).train()
        optim = Ranger(list(E2_trainable.parameters()), lr=0.001)

        progress_bar = tqdm(range(iter_num), desc=f'Runtime E optim')
        for _ in tqdm(progress_bar):
            optim.zero_grad()
            w,t = self.create_w_tp_from_img(img_256, img_512, c, E1=self.encoder_frozen, E2=E2_trainable)
            rec_512 = self.synthesise_from_tp(input_w=w, input_tp=t, pose=c, return_triplanes=False)['image']
            
            lpips_loss = self.metrics.calc_lpips(rec_512, img_512)
            id_loss = self.metrics.calc_id(rec_512, img_512, img_512)
            loss = 1.0*lpips_loss + 0.8*id_loss

            loss.backward()
            optim.step()
            progress_bar.set_postfix(loss=loss.item())

        E2_trainable = E2_trainable.requires_grad_(False).eval()  
        return E2_trainable
    
    def edit_demo(self, input_base_dir, src_name, dst_name, edit_label, runtime_optim):
        """
        Performs reference-based edit, from src_name to dst_name with attribute edit_label.
        Returns concatenated source, destination, and edited PIL images.
        """
        ## Load imgs from paths and camera matrices from labels.json
        def read_img(base_path, img_name):
            return torch.from_numpy(cv2.imread(os.path.join(base_path, img_name))[:,:,[2,1,0]].transpose(2,0,1)).unsqueeze(0).type(torch.float32).to(self.device) / 127.5 - 1

        img_src = read_img(input_base_dir, src_name)
        img_dst = read_img(input_base_dir, dst_name)

        with open(os.path.join(input_base_dir,'dataset.json')) as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        c_src = torch.from_numpy(np.array(labels[src_name])).unsqueeze(0).type(torch.float32).to(self.device)
        c_dst = torch.from_numpy(np.array(labels[dst_name])).unsqueeze(0).type(torch.float32).to(self.device)

        ## Encoding
        if runtime_optim:
            E2_src_tuned = self.runtime_E_optim(self.to_256(img_src), self.to_512(img_src), c_src)
            E2_dst_tuned = self.runtime_E_optim(self.to_256(img_dst), self.to_512(img_dst), c_dst)
        
        w_src, t_src = self.create_w_tp_from_img(self.to_256(img_src), self.to_512(img_src), c_src, 
                                                 E1=self.encoder_frozen,
                                                 E2=E2_src_tuned if runtime_optim else self.encoder_second_stage_frozen)
        w_dst, t_dst = self.create_w_tp_from_img(self.to_256(img_dst), self.to_512(img_dst), c_dst, 
                                                 E1=self.encoder_frozen,
                                                 E2=E2_dst_tuned if runtime_optim else self.encoder_second_stage_frozen)

        ## Editing
        w_fused, t_fused = self.reference_tp_edit(edit_label=edit_label, 
                                                  w_src=w_dst if edit_label == 'hair' else w_src, 
                                                  t_src=t_dst if edit_label == 'hair' else t_src, 
                                                  w_dst=w_src if edit_label == 'hair' else w_dst, 
                                                  t_dst=t_src if edit_label == 'hair' else t_dst)
        
        ## Rendering
        edited = self.synthesise_from_tp(input_w=w_fused, input_tp=t_fused, pose=c_dst, return_triplanes=False)['image']
        src_enc = self.synthesise_from_tp(input_w=w_src, input_tp=t_src, pose=c_src, return_triplanes=False)['image']
        dst_enc = self.synthesise_from_tp(input_w=w_dst, input_tp=t_dst, pose=c_dst, return_triplanes=False)['image']
        orig_pose_renders = torch.cat([self.to_256(img_src), self.to_256(src_enc), self.to_256(img_dst), self.to_256(dst_enc), self.to_256(edited)], dim=-1)

        novel_render_list = []
        for pose in self.novel_pose_list_visualize:
            edited_novel = self.synthesise_from_tp(input_w=w_fused, input_tp=t_fused, pose=pose, return_triplanes=False)['image']
            novel_render_list.append(self.to_256(edited_novel))
        novel_pose_renders = torch.cat(novel_render_list, dim=-1)
        
        # Row 1 : Src original, src encoded, dst original, dst encoded, edited image rendered from original pose
        # Row 2 : Edited from novel poses 
        all_renders = torch.cat([orig_pose_renders, novel_pose_renders], dim=-2)
        torchvision.utils.save_image(all_renders, os.path.join(self.outdir,f'{src_name}_{dst_name}_{edit_label}.png'), normalize=True, value_range=(-1,1))
        return torchvision.transforms.functional.to_pil_image(torch.clip(0.5*(all_renders+1).squeeze(),0,1))