import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

"""
MINIMAL FACE SWAP - DIVIDE AND CONQUER APPROACH

Phase 1: Verify identity encoder works
Phase 2: Verify we can reconstruct from ID alone (sanity check)
Phase 3: Add attributes gradually
"""

IMG_SIZE = 128
ID_DIM = 512
BATCH_SIZE = 16
NUM_EPOCHS = 200
NUM_IDENTITIES = 1298


class FaceSwapDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.identity_to_images = {}
        self.all_images = []
        
        for img_path in glob.glob(os.path.join(img_dir, '*.png')):
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
        img_path_src, person_id_src = self.all_images[idx]
        img_src = Image.open(img_path_src).convert("RGB")
        
        # Same person, different image
        same_person_imgs = self.identity_to_images[person_id_src]
        if len(same_person_imgs) > 1:
            img_path_same = np.random.choice([p for p in same_person_imgs if p != img_path_src])
        else:
            img_path_same = img_path_src
        img_same = Image.open(img_path_same).convert("RGB")
        
        # Different person
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


# ============================================================================
# PHASE 1: IDENTITY ENCODER - Must produce good identity embeddings
# ============================================================================

class SimpleIdentityEncoder(nn.Module):
    """
    Just use pre-trained face recognition. 
    We KNOW this works for identity.
    """
    def __init__(self):
        super(SimpleIdentityEncoder, self).__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        # Remove classification head, keep features
        self.backbone.logits = nn.Identity()
        
    def forward(self, x):
        # Returns 512-dim identity embedding
        return self.backbone(x)


# ============================================================================
# PHASE 2: SIMPLE GENERATOR - Can it use identity at all?
# ============================================================================

class AdaINLayer(nn.Module):
    """Adaptive Instance Normalization - injects identity at each scale"""
    def __init__(self, num_features, id_dim=512):
        super(AdaINLayer, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # Learn scale and shift from identity
        self.fc = nn.Linear(id_dim, num_features * 2)
    
    def forward(self, x, id_code):
        # Normalize features
        normalized = self.norm(x)
        
        # Get style parameters from identity
        style = self.fc(id_code)
        style = style.view(style.size(0), -1, 1, 1)
        scale, shift = style.chunk(2, dim=1)
        
        # Apply identity style
        return scale * normalized + shift


class ImprovedGenerator(nn.Module):
    """
    Generator with multi-scale identity injection.
    Identity is injected at EVERY decoder layer via AdaIN.
    """
    def __init__(self):
        super(ImprovedGenerator, self).__init__()
        
        # Encoder - down to 8x8
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 128 -> 64
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), # 64 -> 32
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), # 32 -> 16
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1), # 16 -> 8
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Bottleneck - inject identity here
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )
        self.adain_bottleneck = AdaINLayer(512)
        
        # Decoder with identity injection at each layer
        self.dec1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)  # 8 -> 16
        self.adain1 = AdaINLayer(512)
        
        self.dec2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 16 -> 32
        self.adain2 = AdaINLayer(256)
        
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 32 -> 64
        self.adain3 = AdaINLayer(128)
        
        self.dec4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 64 -> 128
        self.adain4 = AdaINLayer(64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, img, id_code):
        # Encode
        e1 = self.enc1(img)    # [B, 64, 64, 64]
        e2 = self.enc2(e1)     # [B, 128, 32, 32]
        e3 = self.enc3(e2)     # [B, 256, 16, 16]
        e4 = self.enc4(e3)     # [B, 512, 8, 8]
        
        # Bottleneck with identity
        b = self.bottleneck(e4)
        b = self.adain_bottleneck(b, id_code)
        
        # Decode with identity injection at each step
        d1 = F.relu(self.dec1(b))
        d1 = self.adain1(d1, id_code)
        
        d2 = F.relu(self.dec2(d1))
        d2 = self.adain2(d2, id_code)
        
        d3 = F.relu(self.dec3(d2))
        d3 = self.adain3(d3, id_code)
        
        d4 = F.relu(self.dec4(d3))
        d4 = self.adain4(d4, id_code)
        
        output = self.final(d4)
        return output


# ============================================================================
# VGG PERCEPTUAL LOSS - for better facial features
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better facial feature matching"""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision import models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, input, target):
        # Normalize
        input = (input * 0.5 + 0.5 - self.mean) / self.std
        target = (target * 0.5 + 0.5 - self.mean) / self.std
        
        loss = 0
        x, y = input, target
        
        for layer in [self.slice1, self.slice2, self.slice3]:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        
        return loss


# ============================================================================
# PHASE 3: DISCRIMINATOR - Simple but effective
# ============================================================================

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 8, 1, 0),
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def compute_identity_similarity(encoder, img1, img2):
    """Compute cosine similarity between two images"""
    with torch.no_grad():
        id1 = encoder(img1)
        id2 = encoder(img2)
        id1 = F.normalize(id1, dim=1)
        id2 = F.normalize(id2, dim=1)
        sim = F.cosine_similarity(id1, id2).mean()
    return sim.item()


def debug_phase(epoch, encoder, generator, dataloader, device):
    """
    Debug: Check if identity is actually being used
    """
    encoder.eval()
    generator.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        img_A = batch['source'][:4].to(device)
        img_B = batch['diff_id'][:4].to(device)
        
        # Get identity codes
        id_A = encoder(img_A)
        id_B = encoder(img_B)
        
        # Test 1: Same image, same ID -> should reconstruct well
        recon_A = generator(img_A, id_A)
        
        # Test 2: Image A with ID B -> should look like B
        swap_AB = generator(img_A, id_B)
        
        # Test 3: Image B with ID A -> should look like A
        swap_BA = generator(img_B, id_A)
        
        # Compute identity similarities
        sim_recon = compute_identity_similarity(encoder, img_A, recon_A)
        sim_swap_AB = compute_identity_similarity(encoder, img_B, swap_AB)
        sim_swap_BA = compute_identity_similarity(encoder, img_A, swap_BA)
        
        print(f"\n[DEBUG Epoch {epoch}]")
        print(f"  Reconstruction similarity (should be HIGH): {sim_recon:.4f}")
        print(f"  Swap A->B similarity with B (should be HIGH): {sim_swap_AB:.4f}")
        print(f"  Swap B->A similarity with A (should be HIGH): {sim_swap_BA:.4f}")
        
        # Save visualization
        grid = torch.cat([
            img_A, img_B,           # Row 1: Original A, B
            recon_A, swap_AB,       # Row 2: Recon A, Swap A->B (should look like B)
            swap_BA, img_B          # Row 3: Swap B->A (should look like A), Original B
        ], dim=0)
        
        save_image(grid, f'debug_epoch_{epoch}.png', nrow=4, normalize=True, value_range=(-1, 1))
    
    encoder.train()
    generator.train()


# ============================================================================
# TRAINING
# ============================================================================

def train():
    print("="*80)
    print("MINIMAL FACE SWAP - DEBUGGING VERSION")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = FaceSwapDataset(
        img_dir="C:\\Users\\wangs\\Downloads\\Octoswap\\training\\cropped128\\",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=4, drop_last=True)
    
    # Models
    print("Initializing models...")
    encoder = SimpleIdentityEncoder().to(device)
    generator = ImprovedGenerator().to(device)  # Changed from MinimalGenerator
    discriminator = SimpleDiscriminator().to(device)
    
    # Freeze encoder (we trust pre-trained features)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    print("Identity encoder: FROZEN (using pre-trained)")
    print("Generator parameters:", sum(p.numel() for p in generator.parameters()))
    
    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Losses
    criterion_L1 = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_perceptual = VGGPerceptualLoss().to(device)  # NEW
    
    # UPDATED loss weights - add perceptual
    lambda_recon = 10.0   # Reconstruction loss
    lambda_perc = 3.0     # NEW: Perceptual loss for better features
    lambda_id = 50.0      # VERY HIGH - identity is most important
    lambda_adv = 1.0      # Adversarial
    
    print("\nLoss weights:")
    print(f"  Reconstruction: {lambda_recon}")
    print(f"  Perceptual: {lambda_perc}")
    print(f"  Identity: {lambda_id}")
    print(f"  Adversarial: {lambda_adv}")
    
    # Training history
    history = {
        'g_loss': [], 'd_loss': [], 'id_loss': [], 'recon_loss': []
    }
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    for epoch in range(NUM_EPOCHS):
        generator.train()
        discriminator.train()
        
        epoch_stats = {'g': 0, 'd': 0, 'id': 0, 'recon': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            img_A = batch['source'].to(device)
            img_B = batch['diff_id'].to(device)
            batch_size = img_A.size(0)
            
            # Labels
            real_label = torch.ones(batch_size, 1, 1, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, 1, 1, device=device)
            
            # Get identity codes (frozen encoder)
            with torch.no_grad():
                id_A = encoder(img_A)
                id_B = encoder(img_B)
            
            # ==================== TRAIN GENERATOR ====================
            opt_G.zero_grad()
            
            # Task 1: Reconstruct A from (A, id_A)
            recon_A = generator(img_A, id_A)
            loss_recon_A = criterion_L1(recon_A, img_A)
            loss_perc_A = criterion_perceptual(recon_A, img_A)  # NEW
            
            # Task 2: Swap - Generate B-like image with A's identity
            # Input: B's structure, A's identity
            # Output: Should have A's face
            swap_BA = generator(img_B, id_A)
            
            # Identity loss: swap_BA should match A's identity
            id_emb_A = encoder(img_A)
            id_emb_swap = encoder(swap_BA)
            id_emb_A = F.normalize(id_emb_A, dim=1)
            id_emb_swap = F.normalize(id_emb_swap, dim=1)
            loss_id = 1 - F.cosine_similarity(id_emb_A, id_emb_swap).mean()
            
            # NEW: Also add L2 identity loss for stronger gradient
            loss_id_l2 = F.mse_loss(id_emb_A, id_emb_swap)
            
            # Task 3: Opposite swap for cycle
            swap_AB = generator(img_A, id_B)
            recon_B = generator(img_B, id_B)
            loss_recon_B = criterion_L1(recon_B, img_B)
            loss_perc_B = criterion_perceptual(recon_B, img_B)  # NEW
            
            # Adversarial
            pred_fake = discriminator(swap_BA)
            loss_adv_g = criterion_adv(pred_fake, real_label)
            
            # Total generator loss
            loss_g = (
                lambda_recon * (loss_recon_A + loss_recon_B) / 2 +
                lambda_perc * (loss_perc_A + loss_perc_B) / 2 +  # NEW
                lambda_id * (loss_id + 0.5 * loss_id_l2) +  # Combined identity losses
                lambda_adv * loss_adv_g
            )
            
            loss_g.backward()
            opt_G.step()
            
            # ==================== TRAIN DISCRIMINATOR ====================
            opt_D.zero_grad()
            
            pred_real = discriminator(img_A)
            loss_d_real = criterion_adv(pred_real, real_label)
            
            pred_fake = discriminator(swap_BA.detach())
            loss_d_fake = criterion_adv(pred_fake, fake_label)
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            loss_d.backward()
            opt_D.step()
            
            # Stats
            epoch_stats['g'] += loss_g.item()
            epoch_stats['d'] += loss_d.item()
            epoch_stats['id'] += loss_id.item()
            epoch_stats['recon'] += loss_recon_A.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx+1:4d}/{len(dataloader)} | "
                      f"G: {loss_g.item():.4f} | D: {loss_d.item():.4f} | "
                      f"ID: {loss_id.item():.4f} | Recon: {loss_recon_A.item():.4f}")
        
        # Epoch summary
        n = len(dataloader)
        for k in epoch_stats:
            epoch_stats[k] /= n
            history[f'{k}_loss'].append(epoch_stats[k])
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS} SUMMARY")
        print(f"  Generator Loss:      {epoch_stats['g']:.4f}")
        print(f"  Discriminator Loss:  {epoch_stats['d']:.4f}")
        print(f"  Identity Loss:       {epoch_stats['id']:.4f} {'[GOOD]' if epoch_stats['id'] < 0.3 else '[NEEDS WORK]'}")
        print(f"  Reconstruction Loss: {epoch_stats['recon']:.4f}")
        print(f"{'='*80}\n")
        
        # Debug every 5 epochs
        if (epoch + 1) % 5 == 0:
            debug_phase(epoch + 1, encoder, generator, dataloader, device)
        
        # Checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'history': history
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    # Final save
    torch.save({
        'generator': generator.state_dict(),
        'history': history
    }, 'final_model.pth')
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['g_loss'], label='Generator')
    plt.plot(history['d_loss'], label='Discriminator')
    plt.legend()
    plt.title('Adversarial Losses')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['id_loss'])
    plt.title('Identity Loss (lower = better)')
    plt.xlabel('Epoch')
    plt.axhline(y=0.3, color='r', linestyle='--', label='Target')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['recon_loss'])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


# ============================================================================
# TEST
# ============================================================================
def test(source_img_path, target_img_path, output_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder = SimpleIdentityEncoder().to(device)
    generator = ImprovedGenerator().to(device)
    
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    
    encoder.eval()
    generator.eval()
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    source = transform(Image.open(source_img_path).convert('RGB')).unsqueeze(0).to(device)
    target = transform(Image.open(target_img_path).convert('RGB')).unsqueeze(0).to(device)
    
    # Swap
    with torch.no_grad():
        source_id = encoder(source)
        result = generator(target, source_id)
    
    save_image(result, output_path, normalize=True, value_range=(-1, 1))
    print(f"Swapped face saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser, cli input arguments for swap.py')
    parser.add_argument('-m', '--mode', help="Running mode (train / test)")
    parser.add_argument('-w', '--weight', help="Input path for the faceswap model")
    parser.add_argument('-s', '--swap_face', help="Image path for the cropped (and aligned) swap / source face")
    parser.add_argument('-i', '--input_target', help="Image path for the cropped (and aligned) input / target face")
    parser.add_argument('-i', '--output_recon', help="Image path for the output / reconstructed face")
    cliargs = parser.parse_args()

    srcimg = cv2.imread(cliargs.swap_face)    
    if cliargs.mode == 'train':
        train()
    elif cliargs.mode == 'test':
        test(cliargs.swap_face, cliargs.input_target, cliargs.output_recon, cliargs.weight)
    elif cliargs.mode == 'evaluate':
        print('To be implemented')
        pass

# Future improvements 

# 1. Fine-tune the balance (optional)
# lambda_id = 40.0      # Reduce slightly if faces look too "forced"
# lambda_perc = 5.0     # Increase for sharper features
# lambda_recon = 8.0    # Reduce to allow more identity freedom

# 2. Add skip connections (for better attribute preservation)
# Modify the generator to include U-Net style skip connections from encoder to decoder at matching resolutions.
# 3. Train longer
# Your results at epoch 200 are good. Training to 300-400 epochs might give even cleaner results.