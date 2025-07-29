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
    model.eval() #Set to inference mode instead of training, so no BatchNorm or Dropout layers are active
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
    
    # Extract last hidden state and take the [CLS] token embedding
    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state[:, 0, :]
    
    # Convert to numpy array
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
    #Load test and training sets
    test_df = pd.read_csv("Testing Set.csv", dtype=str)
    train_df = pd.read_csv("Training Set.csv", dtype=str)

    #Create image paths
    test_image_paths = [os.path.join("images", f"image_{img_num}.jpg") 
                        for img_num in test_df["Image #"]]
    
    train_image_paths = [os.path.join("images", f"image_{img_num}.jpg")
                         for img_num in train_df["Image #"]]
    
    # Load model once to reuse
    model, processor, device = load_model()
    
    # Generate embeddings
    test_embeddings = generate_embeddings_batch(
        test_image_paths, model, processor, device)
    
    train_embeddings = generate_embeddings_batch(
        train_image_paths, model, processor, device)
    
    # Save embeddings
    save_embeddings(test_embeddings, "test_embeddings.pkl")
    save_embeddings(train_embeddings, "train_embeddings.pkl")

if __name__ == "__main__":
    main()