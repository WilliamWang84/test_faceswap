import cv2
import glob
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader, Dataset 
from torchvision import models
from torchvision import transforms
from torchvision.utils import save_image


# --- Model Constants ---
IMG_CHANNELS = 3
IMG_SIZE = 128
ID_EMBEDDING_DIM = 512
ATTR_EMBEDDING_DIM = 512
NUM_EPOCHS = 500
BATCH_SIZE = 32
NUM_IDENTITIES = 1298 # Obtained from helper.RecursiveImgParser.get_unique_count(RecursiveImgParser.parse())
TRAIN_DATASET_DIR = "C:\\Users\\wangs\\Downloads\\test_faceswap\\training\\cropped128\\"

# --- Improved Dataset with Identity Labels ---
class FaceSwapDataset(Dataset):
    """
    Dataset that groups images by identity and provides pairs for training.
    Expects folder structure: img_dir/person_id/image.png
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Group images by identity
        self.identity_to_images = {}
        self.all_images = []
        
        # If using flat structure, extract identity from filename
        for img_path in glob.glob(os.path.join(img_dir, '*.png')):
            # Assume filename format: personID_imageNum.png
            filename = os.path.basename(img_path)
            person_id = filename.split('_')[0] if '_' in filename else '0'
            
            if person_id not in self.identity_to_images:
                self.identity_to_images[person_id] = []
            self.identity_to_images[person_id].append(img_path)
            self.all_images.append((img_path, person_id))
        
        self.identity_list = list(self.identity_to_images.keys())
        print(f"Loaded {len(self.all_images)} images from {len(self.identity_list)} identities")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        # Get source image and identity
        img_path_src, person_id_src = self.all_images[idx]
        img_src = Image.open(img_path_src).convert("RGB")
        
        # Get another image of the same person for self-reenactment
        same_person_imgs = self.identity_to_images[person_id_src]
        img_path_same = np.random.choice(same_person_imgs)
        img_same = Image.open(img_path_same).convert("RGB")
        
        # Get image from different person for cross-identity swap
        other_person_id = np.random.choice([pid for pid in self.identity_list if pid != person_id_src])
        img_path_other = np.random.choice(self.identity_to_images[other_person_id])
        img_other = Image.open(img_path_other).convert("RGB")
        
        if self.transform:
            img_src = self.transform(img_src)
            img_same = self.transform(img_same)
            img_other = self.transform(img_other)
        
        return {
            'source': img_src,
            'same_id': img_same,
            'diff_id': img_other,
            'id_label': int(person_id_src) if person_id_src.isdigit() else hash(person_id_src) % NUM_IDENTITIES
        }


# --- Spectral Normalization ---
def spectral_norm(module, mode=True):
    """Apply spectral normalization to a module"""
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# --- ArcFace Loss for Identity ---
class ArcFaceLoss(nn.Module):
    """
    ArcFace loss with angular margin for better identity discrimination.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # Normalize embedding and weight
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Calculate phi with margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, label)


# --- Improved Identity Encoder ---
class EncoderID(nn.Module):
    """
    Identity encoder with ArcFace head for better identity preservation.
    """
    def __init__(self, num_classes=NUM_IDENTITIES, freeze_backbone=True):
        super(EncoderID, self).__init__()
        
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        backbone_out_features = self.backbone.logits.in_features
        self.backbone.logits = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Identity embedding head
        self.id_head = nn.Sequential(
            nn.Linear(backbone_out_features, ID_EMBEDDING_DIM),
            nn.BatchNorm1d(ID_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ArcFace classification head (for training only)
        self.arcface = ArcFaceLoss(ID_EMBEDDING_DIM, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        id_embedding = self.id_head(features)
        
        if labels is not None and self.training:
            arcface_loss = self.arcface(id_embedding, labels)
            return id_embedding, arcface_loss
        
        return id_embedding, None


# --- Attribute Extractor (Frozen) ---
class AttributeExtractor(nn.Module):
    """
    Pre-trained ResNet-50 for extracting pose, lighting, background features.
    """
    def __init__(self):
        super(AttributeExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Extract features from layer3 (good balance of spatial info and semantics)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.feature_extractor(x)


# --- Attention-based ID Injection ---
class AdaptiveIDInjection(nn.Module):
    """
    Improved ID injection with spatial attention to preserve attributes.
    """
    def __init__(self, in_channels, id_embedding_dim=ID_EMBEDDING_DIM):
        super(AdaptiveIDInjection, self).__init__()
        
        # Transform ID embedding to modulation parameters
        self.mlp = nn.Sequential(
            nn.Linear(id_embedding_dim, id_embedding_dim),
            nn.ReLU(),
            nn.Linear(id_embedding_dim, in_channels * 2)
        )
        
        # Spatial attention to preserve attributes
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_attr, z_id):
        # Generate scale and bias
        params = self.mlp(z_id)
        scale, bias = params.chunk(2, dim=1)
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
        bias = bias.view(bias.shape[0], bias.shape[1], 1, 1)
        
        # Normalize and modulate
        norm = F.instance_norm(x_attr)
        modulated = norm * (1 + scale) + bias
        
        # Apply attention to blend with original
        attn_map = self.attention(x_attr)
        return attn_map * modulated + (1 - attn_map) * x_attr


# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


# --- Improved Generator ---
class Generator(nn.Module):
    """
    U-Net style generator with improved ID injection and residual blocks.
    """
    def __init__(self, in_channels=IMG_CHANNELS, id_embedding_dim=ID_EMBEDDING_DIM):
        super(Generator, self).__init__()

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2)
        )

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # ID injection modules (output channels after injection)
        self.id_inject1 = AdaptiveIDInjection(512, id_embedding_dim)
        self.id_inject2 = AdaptiveIDInjection(256, id_embedding_dim)
        self.id_inject3 = AdaptiveIDInjection(128, id_embedding_dim)
        self.id_inject4 = AdaptiveIDInjection(64, id_embedding_dim)

        # Decoder (input channels account for concatenation)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, 4, 2, 1),  # 512 (injected) + 256 (skip d3)
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, 2, 1),   # 256 (injected) + 128 (skip d2)
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1),   # 128 (injected) + 64 (skip d1)
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, in_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x_attr, z_id):
        # Encoder
        d1 = self.down1(x_attr)      # 64 x 64 x 64
        d2 = self.down2(d1)           # 128 x 32 x 32
        d3 = self.down3(d2)           # 256 x 16 x 16
        d4 = self.down4(d3)           # 512 x 8 x 8
        
        # Bottleneck
        b = self.bottleneck(d4)       # 512 x 8 x 8
        
        # Decoder with ID injection and skip connections
        u1 = self.up1(b)              # 512 x 16 x 16
        u1 = self.id_inject1(u1, z_id)
        u1 = torch.cat([u1, d3], dim=1)  # (512 + 256) = 768 channels
        
        u2 = self.up2(u1)             # 256 x 32 x 32
        u2 = self.id_inject2(u2, z_id)
        u2 = torch.cat([u2, d2], dim=1)  # (256 + 128) = 384 channels
        
        u3 = self.up3(u2)             # 128 x 64 x 64
        u3 = self.id_inject3(u3, z_id)
        u3 = torch.cat([u3, d1], dim=1)  # (128 + 64) = 192 channels
        
        u4 = self.up4(u3)             # 64 x 128 x 128
        u4 = self.id_inject4(u4, z_id)
        
        return self.final_conv(u4)    # 3 x 128 x 128


# --- Multi-Scale Discriminator ---
class MultiScaleDiscriminator(nn.Module):
    """
    Discriminator that operates at multiple scales for better detail.
    """
    def __init__(self, in_channels=IMG_CHANNELS):
        super(MultiScaleDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2, 1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Scale 1 (full resolution)
        self.scale1 = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0)),
            nn.Flatten(),
            nn.Linear(25, 1) # Project flattened features to a single logit
        )
        
        # Scale 2 (half resolution)
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(3, stride=2, padding=1),
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0)),
            nn.Flatten(),
            nn.Linear(25, 1) # Corrected: Project flattened 5x5 features
        )
        
    def forward(self, x):
        out1 = self.scale1(x)
        out2 = self.scale2(x)
        return [out1, out2]


# --- VGG Perceptual Loss ---
class VGGPerceptualLoss(nn.Module):
    """Enhanced perceptual loss using VGG-19."""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:18])) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg[18:27]))# relu4_4
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Weights for different layers
        self.weights = [1.0, 1.0, 1.0, 1.0]

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x, y = input, target
        
        for i, (slice_layer, weight) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4], 
            self.weights
        )):
            x = slice_layer(x)
            y = slice_layer(y)
            loss += weight * F.l1_loss(x, y)
        
        return loss


# --- Enhanced Identity Loss ---
class IdentityLoss(nn.Module):
    """Combined cosine and L2 loss for identity preservation."""
    def __init__(self, alpha=0.5):
        super(IdentityLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, emb1, emb2):
        # Normalize embeddings
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        # Cosine similarity loss
        cos_loss = 1 - F.cosine_similarity(emb1_norm, emb2_norm).mean()
        
        # L2 distance loss
        l2_loss = F.mse_loss(emb1_norm, emb2_norm)
        
        return self.alpha * cos_loss + (1 - self.alpha) * l2_loss


# --- Gradient Penalty ---
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Gradient penalty for WGAN-GP stability."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    gradients_list = []
    for d_out in d_interpolates:
        grad = torch.autograd.grad(
            outputs=d_out,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients_list.append(grad.view(grad.size(0), -1))
    
    gradients = torch.cat(gradients_list, dim=1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# --- Training Function ---
def train():
    print("=== Improved Face Swap GAN Training ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data preparation
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FaceSwapDataset(
        img_dir=TRAIN_DATASET_DIR,
        transform=data_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize models
    encoder_id = EncoderID(num_classes=NUM_IDENTITIES, freeze_backbone=True).to(device)
    generator = Generator().to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    
    # Pre-trained models for loss computation
    face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for param in face_recognition.parameters():
        param.requires_grad = False
    
    attribute_extractor = AttributeExtractor().to(device)
    
    # Optimizers with different learning rates
    optimizer_G = optim.Adam(
        itertools.chain(encoder_id.parameters(), generator.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.5)
    
    # Loss functions
    criterion_identity = IdentityLoss()
    criterion_recon = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss().to(device)
    criterion_attr = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()
    
    # Loss weights
    lambda_id = 10.0
    lambda_recon = 10.0
    lambda_perc = 5.0
    lambda_attr = 5.0
    lambda_arcface = 1.0
    lambda_gp = 10.0
    lambda_cycle = 5.0
    
    # Training history
    g_losses, d_losses = [], []
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            img_source = batch['source'].to(device)
            img_same_id = batch['same_id'].to(device)
            img_diff_id = batch['diff_id'].to(device)
            id_labels = batch['id_label'].to(device)
            
            batch_size = img_source.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # =================== Train Generator ===================
            optimizer_G.zero_grad()
            
            # Encode identity
            z_id_source, arcface_loss = encoder_id(img_source, id_labels)
            z_id_same, _ = encoder_id(img_same_id, None)
            
            # 1. Self-reenactment (same person, different pose)
            recon_same = generator(img_same_id, z_id_source)
            loss_recon = criterion_recon(recon_same, img_source)
            loss_perc_recon = criterion_perceptual(recon_same, img_source)
            
            # 2. Cross-identity swap (A's identity on B's attributes)
            swap_face = generator(img_diff_id, z_id_source)
            
            # Identity preservation
            id_emb_original = face_recognition(img_source)
            id_emb_swapped = face_recognition(swap_face)
            loss_id = criterion_identity(id_emb_swapped, id_emb_original)
            
            # Attribute preservation
            attr_target = attribute_extractor(img_diff_id)
            attr_swapped = attribute_extractor(swap_face)
            loss_attr = criterion_attr(attr_swapped, attr_target)
            
            # Perceptual quality
            loss_perc_swap = criterion_perceptual(swap_face, img_diff_id)
            
            # 3. Cycle consistency
            z_id_swapped, _ = encoder_id(swap_face, None)
            cycle_recon = generator(img_source, z_id_swapped)
            loss_cycle = criterion_recon(cycle_recon, img_source)
            
            # 4. Adversarial loss
            fake_preds = discriminator(swap_face)
            loss_adv_g = sum([criterion_adv(pred, real_labels[:pred.size(0)]) for pred in fake_preds]) / len(fake_preds)
            
            # Combine generator losses
            loss_g = (
                loss_adv_g +
                lambda_recon * loss_recon +
                lambda_id * loss_id +
                lambda_attr * loss_attr +
                lambda_perc * (loss_perc_recon + loss_perc_swap) +
                lambda_cycle * loss_cycle
            )
            
            if arcface_loss is not None:
                loss_g += lambda_arcface * arcface_loss
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(encoder_id.parameters(), max_norm=5.0)
            optimizer_G.step()
            
            # =================== Train Discriminator ===================
            optimizer_D.zero_grad()
            
            # Real images
            real_preds = discriminator(img_source)
            loss_d_real = sum([criterion_adv(pred, real_labels[:pred.size(0)]) for pred in real_preds]) / len(real_preds)
            
            # Fake images
            fake_preds = discriminator(swap_face.detach())
            loss_d_fake = sum([criterion_adv(pred, fake_labels[:pred.size(0)]) for pred in fake_preds]) / len(fake_preds)
            
            # Gradient penalty
            gp = compute_gradient_penalty(discriminator, img_source, swap_face.detach(), device)
            
            loss_d = (loss_d_real + loss_d_fake) / 2.0 + lambda_gp * gp
            
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            optimizer_D.step()
            
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            num_batches += 1
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Log progress
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"G_Loss: {avg_g_loss:.4f} | D_Loss: {avg_d_loss:.4f} | "
              f"ID: {loss_id.item():.4f} | Attr: {loss_attr.item():.4f}")
        
        # Save samples and checkpoints
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                save_image(
                    torch.cat([img_source[:4], img_diff_id[:4], swap_face[:4]], dim=0),
                    f"./result_epoch_{epoch+1}.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1)
                )
            
            checkpoint = {
                'epoch': epoch,
                'encoder_id': encoder_id.state_dict(),
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses
            }
            torch.save(checkpoint, f"./checkpoint_epoch_{epoch+1}.pth")
    
    # Save final results
    np.savetxt("g_losses.txt", np.array(g_losses))
    np.savetxt("d_losses.txt", np.array(d_losses))
    
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('training_losses.png')
    plt.show()
    
    print("\nTraining complete!")


if __name__ == '__main__':
    train()
