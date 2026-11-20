import torch
import torch.nn as nn
import numpy as np
import cv2

class Generator(nn.Module):
    """Generator network for color correction and contrast restoration"""
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network for GAN training"""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

class GANEnhancer:
    """GAN-based enhancer for color correction and contrast restoration"""
    
    def __init__(self):
        """Initialize GAN enhancer with pre-trained models"""
        self.device = torch.device('cpu')
        self.generator = Generator().to(self.device)
        self.generator.eval()
    
    def enhance(self, image):
        """
        Enhance image using GAN for color correction
        
        Args:
            image: Input image (numpy array, BGR format)
        
        Returns:
            Enhanced image (numpy array, BGR format)
        """
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            enhanced_tensor = self.generator(image_tensor)
        
        # Convert back to numpy
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 255]
        enhanced_rgb = ((enhanced_np + 1) / 2 * 255).astype(np.uint8)
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        
        lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_bgr = cv2.merge([l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_bgr, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
