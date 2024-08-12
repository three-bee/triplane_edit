import torch
from models.model_irse import Backbone
from criteria.lpips.lpips import LPIPS
from torch.nn.functional import mse_loss

class Metrics:
    def __init__(self, ir_se_50_path, device):
        self.device = device
        
        self.lpips_module = LPIPS(net_type="alex").to(device).eval()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se")
        self.facenet.load_state_dict(torch.load(ir_se_50_path))
        self.facenet.to(device).eval()
        
        self.facenet_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def extract_facenet_feats(self, x, crop_center=True):
        if crop_center:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.facenet_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def calc_id(self, y_hat, y, x):
        y_hat = self.face_pool(y_hat)
        y = self.face_pool(y)
        x = self.face_pool(x)

        n_samples = x.shape[0]
        x_feats = self.extract_facenet_feats(x, crop_center=True)
        y_feats = self.extract_facenet_feats(y, crop_center=True)  # Otherwise use the feature from there
        y_hat_feats = self.extract_facenet_feats(y_hat, crop_center=True)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append(
                {
                    "diff_target": float(diff_target),
                    "diff_input": float(diff_input),
                    "diff_views": float(diff_views),
                }
            )
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count

    def calc_lpips(self, X, Y):
        X = self.face_pool(X)
        Y = self.face_pool(Y)
        return self.lpips_module(X, Y)

    def calc_mse(self, X, Y):
        X = self.face_pool(X)
        Y = self.face_pool(Y)
        return mse_loss(X, Y)

    def calc_l1(self, X, Y):
        X = self.face_pool(X)
        Y = self.face_pool(Y)
        return torch.abs(X - Y).mean()