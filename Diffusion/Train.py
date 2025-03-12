
import os
from typing import Dict
from PIL import Image, ImageDraw
from PIL import ImageFilter

import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from Diffusion import DADA, Trainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler

class CustomDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.images = sorted(os.listdir(image_root))
        self.masks = sorted(os.listdir(mask_root))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.images[idx])
        mask_path = os.path.join(self.mask_root, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        to_tensor = transforms.ToTensor()
        mask = to_tensor(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    
    dataset = CustomDataset(
        image_root = modelConfig["train_image_dir"],
        mask_root = modelConfig["train_mask_dir"],
        transform=transforms.Compose([
            transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
    
    net_model = nn.DataParallel(net_model).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(modelConfig["BG-De_weight"]), map_location=device))
    
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    trainer = Trainer(net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, masks in tqdmDataLoader:
                optimizer.zero_grad()

                x_0 = images.to(device)
                masks = masks.to(device)

                loss = trainer(x_0, masks).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={"epoch": e, "loss: ": loss.item(), "img shape: ": x_0.shape, "LR": optimizer.state_dict()['param_groups'][0]["lr"]})
        warmUpScheduler.step()

        if e >= modelConfig["epoch"] - 5:
            torch.save(net_model.state_dict(), os.path.join(modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)

    model = nn.DataParallel(model).to(device)

    ckpt = torch.load(os.path.join(modelConfig["BG-De_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    print("model load weight done.")
    model.eval()

    base_image_path = modelConfig["inpaint_image"]
    base_image = Image.open(base_image_path).convert("RGB")
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    base_tensor = base_transform(base_image).unsqueeze(0) 

    mask_image_path = modelConfig["inpaint_mask"]
    mask_image = Image.open(mask_image_path).convert("L")
    blur_radius = 5  
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    mask_tensor = mask_transform(mask_image).unsqueeze(0)  

    sampler = DADA(model, base_tensor, mask_tensor, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["Detection_weight"]).to(device)
    noisyImage = torch.randn([1, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)

    sampledImgs = sampler(noisyImage)
    sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
    save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"], f"DADA.png"))  