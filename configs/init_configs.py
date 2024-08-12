import dnnlib
import yaml
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, default='work_dirs')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to swin config file', default='./configs/swinv2.yaml')
    
    parser.add_argument('--E_ckpt_path', type=str, default='./checkpoints/encoder_FFHQ.pt')
    parser.add_argument('--E2_ckpt_path', type=str, default='./checkpoints/afa_FFHQ.pt')
    parser.add_argument('--Efinetuned_ckpt_path', type=str, default='./checkpoints/encoder_FFHQ_finetuned.pt')
    parser.add_argument('--G_ckpt_path', type=str, default='./checkpoints/ffhqrebalanced512-128.pkl')
    parser.add_argument('--mask_gen_ckpt', type=str, default='./checkpoints/79999_iter.pth')
    parser.add_argument('--ir_se_50_path', type=str, default='./checkpoints/model_ir_se50.pth')

    return parser