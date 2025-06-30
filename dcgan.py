import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from PIL import Image
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch


# -------------------------------
# BDD100K Dataset Loader
# -------------------------------
class BDD100KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


# -------------------------------
# Arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/media/volume/100K/bdd100k_images_100k/bdd100k/images/100k/train/", help="Path to BDD100K images")
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--sample_interval", type=int, default=400)
parser.add_argument("--disc_pth", type=str, default="discriminator_128.pth")
parser.add_argument("--gen_pth", type=str, default="generator_128.pth")
opt = parser.parse_args()
print(opt)

os.makedirs("images", exist_ok=True)
cuda = torch.cuda.is_available()

# -------------------------------
# Weight init
# ------------------------------/media/volume/100K/bdd100k_images_100k/bdd100k/images/100k/train/-
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# -------------------------------
# Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        if opt.img_size == 128:
            self.init_size = 4  # 4x4, 5 upsamplings to 128x128
        elif opt.img_size == 64:
            self.init_size = 2  # 2x2, 5 upsamplings to 64x64
        else:
            self.init_size = max(1, opt.img_size // 32)  # fallback
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 1024 * self.init_size ** 2))

        up_blocks = []
        in_channels = 1024
        out_channels_list = [512, 256, 128, 64, 32]
        for out_channels in out_channels_list:
            up_blocks.append(nn.Upsample(scale_factor=2))
            up_blocks.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
            up_blocks.append(nn.BatchNorm2d(out_channels))
            up_blocks.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        up_blocks.append(nn.Conv2d(in_channels, opt.channels, 3, stride=1, padding=1))
        up_blocks.append(nn.Tanh())
        self.conv_blocks = nn.Sequential(*up_blocks)

    def forward(self, z):
        out = self.l1(z)
        #print(f"generator, Shape before conv_blocks: {out.shape}")
        out = out.view(out.size(0), 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        #print(f"generator, output shape: {img.shape}")  # Debug: print generated image shape
        return img

# -------------------------------
# Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 32, bn=False),       # 128 -> 64
            *discriminator_block(32,64),                            # 64 -> 32
            *discriminator_block(64,128),                           # 32 -> 16
            *discriminator_block(128,256),                          # 16 -> 8
            *discriminator_block(256,512),                          # 8 -> 4
            *discriminator_block(512, 1024),                        # 4 -> 2 (removed for 128x128 input)
        )
        # For 128x128 input, use only 5 downsampling blocks so output is 4x4 -> 2x2
        if opt.img_size == 128:
            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 32, bn=False),   # 128 -> 64
                *discriminator_block(32,64),                        # 64 -> 32
                *discriminator_block(64,128),                       # 32 -> 16
                *discriminator_block(128,256),                      # 16 -> 8
                *discriminator_block(256,512),                      # 8 -> 4
            )
            ds_size = 4  # Output is 4x4 after 5 downsamples for 128x128 input
            adv_in_channels = 512  # Output channels after last block
            self.adv_layer = nn.Sequential(nn.Linear(adv_in_channels * ds_size * ds_size, 1), nn.Sigmoid())  # 512*4*4=8192
        else:
            self.model = nn.Sequential(
                *discriminator_block(opt.channels, 32, bn=False),   # 64/other -> 32
                *discriminator_block(32,64),                        # 32 -> 16
                *discriminator_block(64,128),                       # 16 -> 8
                *discriminator_block(128,256),                      # 8 -> 4
                *discriminator_block(256,512),                      # 4 -> 2
                *discriminator_block(512, 1024),                    # 2 -> 1
            )
            ds_size = opt.img_size // (2 ** 6)
            if ds_size < 1:
                ds_size = 1  # Ensure ds_size is at least 1
            adv_in_channels = 1024  # Output channels after last block
            self.adv_layer = nn.Sequential(nn.Linear(adv_in_channels * ds_size * ds_size, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        #print(f"discriminator, shape after conv_blocks: {out.shape}")  # Print actual output shape for debugging
        out = out.view(out.size(0), -1)
        #print(f"discriminator, flattened shape: {out.shape}")  # Debug flattened shape
        validity = self.adv_layer(out)
        return validity


# -------------------------------
# Setup
# -------------------------------
adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

if os.path.exists(opt.gen_pth):
    print(f"Loading pretrained Generator weights from {opt.img_size/2}x{opt.img_size/2}...")
    gen_dict = generator.state_dict()
    pretrained_gen = torch.load(opt.gen_pth)
    pretrained_gen = {k: v for k, v in pretrained_gen.items() if k in gen_dict and gen_dict[k].shape == v.shape}
    gen_dict.update(pretrained_gen)
    generator.load_state_dict(gen_dict)
else: 
    generator.apply(weights_init_normal)

if os.path.exists(opt.disc_pth):
    print(f"Loading pretrained Discriminator weights from {opt.img_size/2}x{opt.img_size/2}...")
    disc_dict = discriminator.state_dict()
    pretrained_disc = torch.load(opt.disc_pth)
    pretrained_disc = {k: v for k, v in pretrained_disc.items() if k in disc_dict and disc_dict[k].shape == v.shape}
    disc_dict.update(pretrained_disc)
    discriminator.load_state_dict(disc_dict)
else: 
    discriminator.apply(weights_init_normal)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# -------------------------------
# Data Loader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = BDD100KDataset(root_dir=opt.data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# -------------------------------
# Optimizers
# -------------------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# -------------------------------
# Training
# -------------------------------
for epoch in range(opt.n_epochs):
    saved = 0
    for i, (imgs, _) in enumerate(dataloader):


        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # Train Generator
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.size(0), opt.latent_dim))))
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
              f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        batches_done = epoch * len(dataloader) + i
        if i == 0:
            save_image(gen_imgs.data[:25], f"images/epoch_{epoch}.png", nrow=5, normalize=True)
        if epoch == opt.n_epochs - 1 and saved == 0:
            saved=+1
            os.makedirs("checkpoints", exist_ok=True)
            savetime =  datetime.now()
            formatted_time = savetime.strftime("%H-%M-%S")
            torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch}_{formatted_time}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch}_{formatted_time}.pth")

