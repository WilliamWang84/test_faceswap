import argparse
import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from facenet_pytorch import InceptionResnetV1
from helper import RecursiveImgParser, detect_face_in_cvframe, detect_face_in_pilimage
from torchvision import models

IMG_CHANNELS = 3
ID_EMBEDDING_DIM = 512
ATTR_EMBEDDING_DIM = 512
IMG_SIZE = 128
NUM_IDENTITIES = 1298

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


data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    # Apply ImageNet avg and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_swap_ab_image_as_cvimg(a_img, b_img, checkpoint_path='checkpoint_epoch_30.pth'):
    
    # Initialize the models
    netG = Generator()
    netE = EncoderID()

    # Check if a GPU is available and set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint on device: {device}")

    # Load the checkpoint dictionary
    # Use map_location to ensure it loads to the correct device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the state_dict for each network
    netG.load_state_dict(checkpoint['generator'])
    netE.load_state_dict(checkpoint['encoder_id'])

    netG.to(device)
    netE.to(device)

    netG.eval()
    netE.eval()

    tensor_source = data_transform(a_img)
    tensor_target = data_transform(b_img)

    batched_image_src = tensor_source.unsqueeze(0)
    batched_image_dst = tensor_target.unsqueeze(0)

    batched_src_device = batched_image_src.to(device)
    batched_dst_device = batched_image_dst.to(device)

    latent_vec_src_id, _ = netE(batched_src_device, None)
    output_image_b_a = netG(batched_dst_device, latent_vec_src_id)

    output_image_b_a = output_image_b_a.squeeze(0) 
    output_image_b_a = output_image_b_a.permute(1, 2, 0) 
    output_image_b_a = output_image_b_a.cpu().detach()
    output_image_b_a = output_image_b_a.numpy()

    output_image_b_a = (output_image_b_a * 255.0).astype(np.uint8)
    return output_image_b_a

def evaluate_idlosses(swap_face_path=None, ls_target_faces_paths=None, checkpoint_path='checkpoint_epoch_30.pth'):
    ls_losses_b2ba = []
    ls_losses_a2ba = []
    criterion_identity = IdentityLoss()

    face_recognition = InceptionResnetV1(pretrained='vggface2').eval()
    for param in face_recognition.parameters():
        param.requires_grad = False

    if swap_face_path and ls_target_faces_paths:
        srcface = Image.open(swap_face_path)
        for dst_face_path in ls_target_faces_paths:
            dstface = Image.open(dst_face_path)
            outface = get_swap_ab_image_as_cvimg(srcface, dstface, checkpoint_path=checkpoint_path)
            outface = cv2.cvtColor(outface, cv2.COLOR_BGR2RGB)
            outface = Image.fromarray(outface)

            dsrcface = data_transform(srcface)
            ddstface = data_transform(dstface)
            doutface = data_transform(outface)
            bsrcface = dsrcface.unsqueeze(0)            
            bdstface = ddstface.unsqueeze(0)
            boutface = doutface.unsqueeze(0)
            # Measure id distances 
            id_src = face_recognition(bsrcface)
            id_dst = face_recognition(bdstface)
            id_out = face_recognition(boutface)
            loss_id_b2ba = criterion_identity(id_dst, id_out)
            loss_id_a2ba = criterion_identity(id_src, id_out)
            ls_losses_b2ba.append(loss_id_b2ba)
            ls_losses_a2ba.append(loss_id_a2ba)
    
    plt.figure(figsize=(10, 5))
    plt.plot(ls_losses_b2ba, label='Target To Swapped Image ID Loss')
    plt.plot(ls_losses_a2ba, label='Source To Swapped Image ID Loss')
    plt.xlabel('ID')
    plt.ylabel('Loss')
    plt.title('ID Swap Loss')
    plt.legend()
    plt.savefig('idswap_losses.png')
    plt.show()

    cnt_total = 0 + np.finfo(float).eps # prevent div 0
    cnt_success = 0
    for lb2ba, la2ba in zip(ls_losses_b2ba, ls_losses_a2ba):
        # Succeed only when id cost of swap face to swapped faces less than cost of target face to swapped face
        if la2ba < lb2ba:
            cnt_success += 1
        cnt_total += 1

    acc_percent = cnt_success / cnt_total
    print("Evaluated ID swap accuracy (%) is: {:.0%}".format(acc_percent))

if __name__ == '__main__':

    ###################################################################
    # Perform CLI based batch test
    # python test.py --weight <checkpoint_path.pth> --swap_face <src.jpg> --input_dir <d:\\input\\> [--output_dir <d:\\output\\>]
    ###################################################################

    # Define input arguments
    parser = argparse.ArgumentParser(description='Parser, cli input arguments for test.py')
    parser.add_argument('-s', '--swap_face', help="Input file path for the toswap / source face image")
    parser.add_argument('-w', '--weight', help="Input path for the faceswap model")
    parser.add_argument('-i', '--input_dir', help="Input directory path for the input / target faces")
    parser.add_argument('-o', '--output_dir', type=str, default='./', help="Output directory path for the output / swapped faces")
    cliargs = parser.parse_args()

    srcimg = cv2.imread(cliargs.swap_face)
    srcface, _ = detect_face_in_cvframe(cvframe=srcimg) # face_img as PIL or CV

    rip1 = RecursiveImgParser(root_dir=cliargs.input_dir, img_formats=['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp','*.avi','*.mp4','*.mkv'])
    ls_inputs = rip1.parse()    
    for ind, inp in enumerate(ls_inputs):
        print("Processing image ", str(ind+1))
        inpfilename = os.path.basename(inp)
        if inp[-3:] in ['avi','mp4','mkv']:
            # Process as video
            vidcap = cv2.VideoCapture(inp)
            frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))

            output_video_path = os.path.join(cliargs.output_dir, inpfilename)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can choose other codecs like 'MP4V'
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_number = 0
            success, frame = vidcap.read()
            while success:
                # Detect and crop face in frame
                face_img, xywh = detect_face_in_cvframe(cvframe=frame) 
                if face_img:
                    outface = get_swap_ab_image_as_cvimg(srcface, face_img, checkpoint_path=cliargs.weight)
                    outface = cv2.cvtColor(outface, cv2.COLOR_BGR2RGB)
                    frame[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]] = cv2.resize(outface, (xywh[2], xywh[3]))
                    
                    # Write replaced frame to new video
                    out.write(frame)
                # Read the next frame
                success, frame = vidcap.read()
                frame_number += 1
            vidcap.release() # Release the video capture object
            out.release()       

        else:
            # Process as image           
            dstimg = cv2.imread(inp)
            dstface, xywh = detect_face_in_cvframe(cvframe=dstimg) # face_img as PIL
            
            outface = get_swap_ab_image_as_cvimg(srcface, dstface, checkpoint_path=cliargs.weight)
            outface = cv2.cvtColor(outface, cv2.COLOR_BGR2RGB)
            outfilename = os.path.join(cliargs.output_dir, inpfilename)
            dstimg[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]] = cv2.resize(outface, (xywh[2], xywh[3])) 
            cv2.imwrite(outfilename, dstimg)


    ##########################################################################
    # Perform ID switch accuracy check
    # 1. Ensure all swap images and input images are cropped to face region
    ##########################################################################
    # ls_targets = glob.glob('C:\\Users\\wangs\\Downloads\\lfw_funned_test\\ls_targets\\*.png')
    # evaluate_idlosses(swap_face_path='C:\\Users\\wangs\\Downloads\\lfw_funned_test\\Vicente_Fox_12739_001.png', ls_target_faces_paths=ls_targets)
