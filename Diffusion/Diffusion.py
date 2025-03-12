
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import (check_file, check_img_size, non_max_suppression, scale_boxes, box_iou)
from utils.torch_utils import select_device, smart_inference_mode
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS

import numpy as np

def add_noise(image, num_steps, beta_1, beta_T, T):
    device = image.device

    betas = torch.linspace(beta_1, beta_T, T).to(device)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)

    x_t = image
    noise = torch.randn_like(image)
    x_t = (extract(sqrt_alphas_bar, torch.tensor([num_steps], device=device), image.shape) * x_t + extract(sqrt_one_minus_alphas_bar, torch.tensor([num_steps], device=device), image.shape) * noise)
    
    return x_t

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def initialize_model(weights, device, imgsz):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, _ = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))
    return model


class Trainer(nn.Module):
    def __init__(self, ddpm_model, beta_1, beta_T, T):
        super().__init__()

        self.ddpm_model = ddpm_model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, mask):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        predicted_noise = self.ddpm_model(x_t, t)

        masked_predicted_noise = predicted_noise * mask
        masked_actual_noise = noise * mask

        loss = F.mse_loss(masked_predicted_noise, masked_actual_noise, reduction='none')

        return loss

class DADA(nn.Module):
    def __init__(self, model, base_tensor, mask_tensor, beta_1, beta_T, T, detection_model):
        super().__init__()

        self.model = model
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.base_tensor = base_tensor
        self.mask_tensor = mask_tensor

        self.stop_time = 100

        self.weights = detection_model
        self.imgsz = (256, 256)
        self.conf_thres=0.25
        self.iou_thres=0.5
        self.max_det=1000
        self.agnostic_nms=False
        self.augment=False
        self.classes=None
        self.device="cuda:0"

        self.item = 1
        self.alpha = 0.003 
        self.eps = 0.2      

        self.yolo_model = initialize_model(self.weights, self.device, self.imgsz)
        self.yolo_model.eval()

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps)

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T                                    

        noisy_base = self.base_tensor
        noisy_base = noisy_base.to(x_t.device) 

        inverted_mask_tensor = 1 - self.mask_tensor     
        inverted_mask_tensor = inverted_mask_tensor.to(x_t.device)

        self.base_tensor = self.base_tensor.to(x_t.device)
        self.mask_tensor = self.mask_tensor.to(x_t.device)

        for time_step in reversed(range(self.T)):
            print(time_step)
            noisy_image = add_noise(noisy_base, num_steps=time_step, beta_1=self.beta_1, beta_T=self.beta_T, T=1000)  
            noisy_image = noisy_image * self.mask_tensor                                         
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step                     
            
            if(time_step % 1 == 0):
                x_t.requires_grad = True
                ori_x_t = x_t.detach().clone()

                mean, var = self.p_mean_variance(x_t=x_t, t=t)
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t_minus_1  = mean + torch.sqrt(var) * noise                                                 

                labels = torch.tensor([0.0], device = self.device)
                true_x_t_minus_1 = torch.clip(x_t_minus_1 , -1, 1)
                true_x_t_minus_1 = true_x_t_minus_1 * 0.5 + 0.5
                
                loss = nn.MSELoss()

                pred = self.yolo_model(true_x_t_minus_1, augment = self.augment)
                confidence = torch.tensor(0.0, device = self.device, requires_grad=True)

                for box in pred[0][0]:
                    confidence = confidence + box[4]
                
                self.yolo_model.zero_grad()
                self.model.zero_grad()
                
                cost = loss(confidence, labels).to(self.device)
                cost.backward()

                adv_images = ori_x_t + self.alpha * x_t.grad.sign()
                eta = torch.clamp(adv_images - ori_x_t, min=-self.eps, max=self.eps)
                true_x_t = (ori_x_t + eta).detach()

                last_eta = true_x_t - ori_x_t
                x_t = ori_x_t + last_eta

            with torch.no_grad():
                mean, var = self.p_mean_variance(x_t=x_t, t=t)
            if time_step > 100:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t_minus_1  = mean + torch.sqrt(var) * noise * 1.7

            x_t_minus_1  = x_t_minus_1  * inverted_mask_tensor
            x_t_minus_1  = x_t_minus_1  + noisy_image   
            x_t = x_t_minus_1
            
        x_0 = x_t                                                                               
        return torch.clip(x_0, -1, 1)                                                          

