import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# Import model functions
from flowering_dual_model import FloweringDualModel
from embedding import load_model, load_image, generate_embedding

# Paths
image_folder = "images"
filtered_csv = "multimedia_human_observations_only.csv"
output_csv = "image_occurrence_log.csv"
flower_model_path = "trained_flower_classifier.joblib"
silique_model_path = "trained_silique_classifier.joblib"
model_info_path = "trained_dual_classifier_info.pkl"

# Load the DINOv2 model for embedding generation
print("Loading DINOv2 model for embedding generation...")
dinov2_model, processor, device = load_model()

# Load the trained dual classifier
print("Loading trained dual classifier...")
classifier = FloweringDualModel()
classifier.load_model(flower_model_path, silique_model_path, model_info_path)
print(f"Models loaded with thresholds: Flower={classifier.flower_threshold:.3f}, Silique={classifier.silique_threshold:.3f}")

# Load CSV
df_media = pd.read_csv(filtered_csv)

# Create sequential image numbers for each gbifID based on order in CSV
df_media['image_sequence'] = df_media.groupby('gbifID').cumcount() + 1

# Initialize list to hold results
results = []

# Track progress
print(f"Processing {len(df_media)} images...")

for idx, row in tqdm(df_media.iterrows(), total=len(df_media)):
    gbifID = row["gbifID"]
    image_sequence = row["image_sequence"]
    image_file = os.path.join(image_folder, f"{gbifID}_{image_sequence}.jpg")

    # Skip if file doesn't exist
    if not os.path.exists(image_file):
        print(f"Missing image: {image_file}")
        continue

    try:
        # Load and generate embedding
        image = load_image(image_file)
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue
            
        emb = generate_embedding(image, dinov2_model, processor, device)
        if emb is None:
            print(f"Failed to generate embedding for: {image_file}")
            continue

        # Run inference with model and extract results
        prediction = classifier.predict(emb)
        
        # Extract flower and silique predictions
        flower_prob = prediction['flowers']['probability']
        flower_present = int(prediction['flowers']['present'])
        silique_prob = prediction['siliques']['probability']
        silique_present = int(prediction['siliques']['present'])
        
        # Apply flowering rule: Flowering = (flowers=1 AND siliques=0)
        is_flowering = flower_present and not silique_present
        predicted_label = "FL" if is_flowering else "Non-FL"

        # Append to results csv
        results.append({
            "gbifID": gbifID,
            "image_filename": f"{gbifID}_{image_sequence}.jpg",
            "predicted_label": predicted_label,
            "flower_probability": flower_prob,
            "flower_present": flower_present,
            "silique_probability": silique_prob,
            "silique_present": silique_present
        })
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        continue

# Convert to DataFrame for saving
df_results = pd.DataFrame(results)

# Optional: Compute occurrence-level flowering (uncomment if needed)
# df_results['occurrence_flowering'] = df_results.groupby('gbifID')['predicted_label'] \
#                                                  .transform(lambda x: int((x == 'FL').any()))

# Output Results and Save
df_results.to_csv(output_csv, index=False)
print(f"Saved image predictions to {output_csv}")
print(f"Total images processed: {len(df_results)}")
print(f"FL predictions: {(df_results['predicted_label'] == 'FL').sum()}")
print(f"Non-FL predictions: {(df_results['predicted_label'] == 'Non-FL').sum()}")
print(f"Images with flowers: {df_results['flower_present'].sum()}")
print(f"Images with siliques: {df_results['silique_present'].sum()}")
print(f"Images with both: {(df_results['flower_present'] & df_results['silique_present']).sum()}")
print(f"Images with neither: {(~df_results['flower_present'] & ~df_results['silique_present']).sum()}")