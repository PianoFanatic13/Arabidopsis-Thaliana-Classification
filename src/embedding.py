#!/usr/bin/env python3
"""
Embedding Generation Module for Arabidopsis Flowering Analysis

Generates embeddings for flower and silique classification, using DINOv2 Large model (308M parameters).
Processes training and testing images to create 1024-dimensional feature vectors for model training.
"""

import numpy as np
import pandas as pd
import os
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pickle

#Load model and processor
def load_model(model_name="facebook/dinov2-large", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large")
    model.eval()
    model.to(device)

    return model, processor, device



def load_image(image_path):
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_embedding(image, model, processor, device):
    """Generate embedding for a single image"""
    if image is None:
        return None
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last hidden state and take the CLS token embedding
    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state[:, 0, :]
    
    #1024D Embedding Vector
    return embedding.cpu().numpy().flatten()

def generate_embeddings_batch(image_paths, model=None, processor=None, device=None, batch_size=32):
    """Generate embeddings for a list of images"""
    # Load model if not provided
    if model is None or processor is None or device is None:
        model, processor, device = load_model()
    
    embeddings = {}
    for i, img_path in enumerate(image_paths):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        image = load_image(img_path)
        if image is not None:
            embedding = generate_embedding(image, model, processor, device)
            embeddings[img_path] = embedding
    
    return embeddings

def save_embeddings(embeddings, output_path):
    """Save embeddings to a file"""
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {output_path}")

def load_embeddings(input_path):
    """Load embeddings from a file"""
    with open(input_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def main(): 
    #Load all training and testing sets
    test_df = pd.read_csv("../data/Testing Set New.csv", dtype=str)
    train_flower_df = pd.read_csv("../data/Flower Training Set.csv", dtype=str)
    train_silique_df = pd.read_csv("../data/Silique Training Set.csv", dtype=str)

    print(f"Loaded datasets:")
    print(f"  Testing Set: {len(test_df)} samples")
    print(f"  Flower Training Set: {len(train_flower_df)} samples")
    print(f"  Silique Training Set: {len(train_silique_df)} samples")

    # Training and Testing sets use "Image Name" column with "gbifID_#.jpg" format
    test_image_paths = [os.path.join("images", f"{img_name}.jpg") 
                        for img_name in test_df["Image Name"]]
     
    flower_train_image_paths = [os.path.join("images", f"{img_name}.jpg")
                               for img_name in train_flower_df["Image Name"]]
    
    silique_train_image_paths = [os.path.join("images", f"{img_name}.jpg")
                                for img_name in train_silique_df["Image Name"]]
    
    print(f"\nImage paths created:")
    print(f"  Test images: {len(test_image_paths)}")
    print(f"  Flower training images: {len(flower_train_image_paths)}")
    print(f"  Silique training images: {len(silique_train_image_paths)}")
    
    # Check if some sample images exist
    print(f"\nSample image path checks:")
    for i, (dataset, paths) in enumerate([
        ("Testing", test_image_paths[:3]),
        ("Flower Training", flower_train_image_paths[:3]),
        ("Silique Training", silique_train_image_paths[:3])
    ]):
        print(f"  {dataset} samples:")
        for path in paths:
            exists = os.path.exists(path)
            print(f"    {os.path.basename(path)}: {'✓' if exists else '✗'}")
    
    # Load model once to reuse
    print(f"\nLoading DinoV2 model...")
    model, processor, device = load_model()
    print(f"Model loaded on device: {device}")
    
    # Generate embeddings for each dataset
    print(f"\n" + "="*50)
    print("GENERATING EMBEDDINGS")
    print("="*50)
    
    print(f"\n1. Generating testing set embeddings...")
    test_embeddings = generate_embeddings_batch(
        test_image_paths, model, processor, device)
    
    print(f"\n2. Generating flower training set embeddings...")
    flower_train_embeddings = generate_embeddings_batch(
        flower_train_image_paths, model, processor, device)
    
    print(f"\n3. Generating silique training set embeddings...")
    silique_train_embeddings = generate_embeddings_batch(
        silique_train_image_paths, model, processor, device)
    
    # Save embeddings
    print(f"\n" + "="*50)
    print("SAVING EMBEDDINGS")
    print("="*50)
    
    save_embeddings(test_embeddings, "test_embeddings_new.pkl")
    save_embeddings(flower_train_embeddings, "flower_train_embeddings.pkl")
    save_embeddings(silique_train_embeddings, "silique_train_embeddings.pkl")
    
    # Summary
    print(f"\n" + "="*50)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*50)
    print(f"Test embeddings: {len(test_embeddings)} images")
    print(f"Flower train embeddings: {len(flower_train_embeddings)} images")
    print(f"Silique train embeddings: {len(silique_train_embeddings)} images")
    
    print(f"\nFiles saved:")
    print(f"  test_embeddings_new.pkl")
    print(f"  flower_train_embeddings.pkl")
    print(f"  silique_train_embeddings.pkl")
    
    # Check for any missing images
    total_expected = len(test_image_paths) + len(flower_train_image_paths) + len(silique_train_image_paths)
    total_generated = len(test_embeddings) + len(flower_train_embeddings) + len(silique_train_embeddings)
    
    if total_generated < total_expected:
        print(f"WARNING: {total_expected - total_generated} images could not be processed")
        print(f"Expected: {total_expected}, Generated: {total_generated}")
    else:
        print(f"\nAll images processed successfully!")

if __name__ == "__main__":
    main()
