import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

class CNNEnhancer:
    """CNN-based image enhancement for noise removal and clarity improvement"""
    
    def __init__(self):
        """Initialize CNN enhancer with autoencoder model"""
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a convolutional autoencoder for image enhancement"""
        inputs = keras.Input(shape=(None, None, 3))
        
        # Encoder
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        # Decoder
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def enhance(self, image):
        """
        Enhance image by removing noise and improving clarity
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Enhanced image (numpy array)
        """
        # Normalize image to [0, 1]
        image_normalized = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        enhanced_batch = self.model.predict(image_batch, verbose=0)
        
        # Remove batch dimension and convert back to [0, 255]
        enhanced = (enhanced_batch[0] * 255).astype(np.uint8)
        
        # Apply bilateral filter for additional denoising
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
