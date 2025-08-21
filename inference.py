import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# Import model functions
from rf_calibrated_model import FloweringClassifier, load_embedding_dict
from embedding import load_model, load_image, generate_embedding

# Paths
image_folder = "images"
filtered_csv = "multimedia_human_observations_only.csv"
output_csv = "image_occurrence_log.csv"
model_path = "trained_flowering_model.joblib"

# Load the DINOv2 model for embedding generation
print("Loading DINOv2 model for embedding generation...")
dinov2_model, processor, device = load_model()

# Load the trained flowering classifier
print("Loading trained flowering classifier...")
classifier = FloweringClassifier()
classifier.load_model(model_path)
print(f"Model loaded with threshold: {classifier.best_threshold:.3f}")

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
        predicted_label = prediction['predicted_class']
        predicted_prob = prediction['probability_FL']

        # Append to results csv
        results.append({
            "gbifID": gbifID,
            "image_filename": f"{gbifID}_{image_sequence}.jpg",
            "predicted_label": predicted_label,
            "predicted_prob": predicted_prob
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