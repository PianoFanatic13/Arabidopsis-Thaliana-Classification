# Arabidopsis Thaliana Flowering Classification using DINOv2

## Overview
This repository contains the models and analysis used to classify flowering stages of the *Arabidopsis thaliana* from field images using self-supervised visual representations derived from Meta's DINOv2 model. The project focuses on distinguishing flowering and non-flowering states, including the presence of siliques, to support phenological analysis and downstream biological research. This project was done under the guidance of USDA Researcher Dr. Xianran Li

The work emphasizes robust feature extraction from limited labeled data and evaluates the effectiveness of modern self-supervised vision transformers for fine-grained plant phenology classification.

## Research Motivation
The *Arabidopsis Thaliana L. Heynh* is one of the most widely studied organisms in plant biology. It serves as a model species, allowing insights gained from studying that plant to be directly applied to research of other more complex or less studied plants. And the flowering time is a key trait for understanding plant development and environmental response. While flowering phenology has been extensively studied in controlled laboratory and common garden experiments, much less is known about flowering patterns across large geographic areas, in the wild. 

At the same time, biodiversity platforms such as GBIF, which compile millions of georeferenced plant observations and images collected, many of which are contributed through citizen science efforts, like the iNaturalist platform. These images provide the opportunity to study phenology outside of controlled environments, but are also collected under variable conditions and without too much standardization for imaging protocols. As a result, these images introduce challenges like inconsistency in image quality, partial and obfuscated views of plants, and ambiguity between closely related reproductive stages, like early flowering and post-flowering. This variability makes phenological inference from images alone difficult, and motivates the structure-aware approach taken in this project.

This project explores whether modern computer vision techniques can be used to infer flowering stages of the *Arabidopsis thaliana* from unconstrained field images, at scale. Manually labeling flowering stages for a large collection of images is time-consuming and does not scale well, especially for phenological analysis at the continental-scale, which this project aims to develop. To address this bottleneck, we rely on self-supervised vision models that leverage visual structure learned from large amounts of unlabeled image data, reducing the need for extensive and tedious manual annotation of images. Inspired by recent work such as the FLORIST AI, this approach aims to extract biologically meaningful signals like flowering status from noisy, real-world images in a way that's reliable and robust. 


## Datasets

### Data Overview

The primary data for this project comes from the [GBIF](https://www.gbif.org/), which aggregates millions of georeferenced plant observations and images, many contributed through citizen science initiatives like iNaturalist. Each observation in GBIF is assigned a unique `gbifID`, which corresponds to a single occurrence of a plant species at a given time and location. An occurrence may have one or more associated images, often capturing multiple perspectives of the same plant or population. Of the occurrences that did have multiple images, most had around 3-4 images.

**Across the dataset:**
- Number of occurrences: 13,256
- Number of images: 22,398
- Average images per occurrence: ~1.7

GBIF Occurrence Download [https://doi.org/10.15468/dl.2w3rzw](https://doi.org/10.15468/dl.2w3rzw)

### Data Filtering

To ensure data quality and relevance for inference, the following filters were applied: 
- Basis of record: Human Observation  
- Continent: Europe  
- Has coordinates: True  
- No geospatial issues  
- Media type: Image  
- Taxon: *Arabidopsis thaliana*

> ### Note on Data Splitting



## Model and Evaluation
This project infers flowering phenology from noisy, field-collected images by using self-supervision visual representations of images. 


### Visual Feature Extraction
Images are embedded using a pretrained DINOv2 ViT-L/14 vision transformer, used as a frozen feature extractor. Inputted images vary in resolution and are resized to a fixed resolution and passed through, becoming a 1024-dimensional embedding per an image, computed via the pooling of the final token representations of the image. With the image embeddings, flowering inference is done via light weight downstream classification, requiring no fine-tuning for the transformer model. DINOv2 was selected for it's ability to capture the visual structure of plants in a robust manner, using representations learned through initial large-scale pretraining of the model. This notion comes in handy especially for the specific dataset, highly variable images that are often noisy and occluded. 

A central goal of the project was to be able to infer flowering status without extensive manual annotation of images. Traditional supervised approaches based on CNNs typically require large labeled datasets, and substantial time for training efforts. In contrast, DINOv2 enables accurate downstream classification with relatively small labeled datasets and simple classifiers, addressing the bottleneck mentioned in the project motivation, while still preserving accuracy in classification. 

### Classification Strategy
In this project, we use a very strict definition for what was considered flowering for a plant. Plants were classified as flowering if they were in the early stages of flowering, right from the time the first anthers were visible, to when it starts reproducing and siliques, or seed pods, were visible, a stage plants would only be in for a few weeks at most. From this we can derive simple visual cues as to help classify flowering stage: 
- No flower petals or siliques present -> Non-Flowering
- Flower petals present and no siliques -> Flowering
- Siliques present at all -> Post-flowering

To help spot these two visual traits in plant images, the pipeline uses two independent binary classifiers for images, one for petal presence, and the other for silique presence. Flowering = (petals present) AND (siliques absent)

### Training Setup
A small set of images labeled for petal and silique presence were used for training and testing. To prevent data leaks, dataset splits are group-aware in regards to Occurrence ID. Since each occurrence may contain multiple images, during training and testing, only one image per occurrence was included in each split, ensuring similar images of the same plant don't appear across both the train and test sets. The training sets for both the silique and petal classifiers were sampled with balanced classes to address the issue of class imbalance across the overall image data. The testing set was sampled randomly and is imbalanced to allow us to get a better sense of how the models would perform on the real-world, imbalanced dataset for inference. 

### Downstream Classifiers
Random Forest classifiers were used for the task of petal and silique detection, executing on the DINOv2 generated embeddings of the images. And was used primarily due to:
- Strong performance on tabular representations
- Robustness to smaller datasets and less prone to overfitting on them
- Ability to model non-linear relationships well, since flower and silique visual cues aren't linearly independent
- Minimal need for extensive hyperparameter tuning and other scaling/normalization requirements

Alternative model approaches were tested for classification, including:
- Support Vector Machines (SVMs)
- Gradient boosting models (XGBoost)
- Dimensionality reduction (PCA & tSNE) prior to classification
- SMOTE-based oversampling for class imbalance
- Probability calibration techniques, like sigmoid regression

Across these experiments, improvements were marginal. The primary performance limitation was feature separability, particularly for silique detection, rather than the choice of classifier. This reinforced the design decision to prioritize representation quality and biologically grounded inference over model complexity.

### Evaluation Metrics and Thresholding
Model performance was evaluated using:

- Precision
- Recall
- F1-score

Initial model selection was guided by AUC-ROC curves to get a good basis of where the optimal thresholds should lie. However, final decision thresholds were adjusted to prioritize high precision over recall. This choice reflects the downstream goal of phenological analysis. False positive flowering predictions can introduce systematic errors when analyzing inference results for geographic trends, whereas false negatives primarily reduce sensitivity and reduce the overall amount of data to base these analyses on. As a result, thresholds were tuned to minimize incorrect flowering assignments, even at the cost of lower recall, as it was more important that we didn't base trends of incorrect classifications of plants.

### Dataset Distributions

**Flower and Silique Classifications**

| Split | Task Focus | Absent (Class 0) | Present (Class 1) | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | Flower | 20 | 20 | 40 |
| **Training** | Silique | 36 | 34 | 70 |
| **Testing** | Flower | 64 | 336 | 400 |
| **Testing** | Silique | 103 | 297 | 400 |

**Overall Flowering Status**

| Split | Task Focus | Non-Flowering | Flowering | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Testing** | Overall Status | 355 | 45 | 400 |

---

### Performance Summary

#### Flower Classification

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Absent (0)** | 0.93 | 0.78 | 0.85 | 64 |
| **Present (1)** | 0.96 | 0.99 | 0.97 | 336 |

**Overall Accuracy:** 0.9550     
**Weighted Avg (P/R/F1):** 0.95 / 0.95 / 0.95

**Confusion Matrix (Actual × Predicted)**

| Actual \\ Predicted | Absent (0) | Present (1) |
| :--- | :--- | :--- |
| **Absent (0)** | 50 | 14 |
| **Present (1)** | 4 | 332 |

---

#### Silique Classification

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Absent (0)** | 1.00 | 0.65 | 0.79 | 103 |
| **Present (1)** | 0.89 | 1.00 | 0.94 | 297 |

**Overall Accuracy:** 0.9100    
**Weighted Avg (P/R/F1):** 0.92 / 0.91 / 0.90 

**Confusion Matrix (Actual × Predicted)**

| Actual \\ Predicted | Absent (0) | Present (1) |
| :--- | :--- | :--- |
| **Absent (0)** | 67 | 36 |
| **Present (1)** | 0 | 297 |

---

#### Overall Flowering Status

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Non-Flowering (0)** | 0.95 | 0.99 | 0.97 | 355 |
| **Flowering (1)** | 0.93 | 0.56 | 0.69 | 45 |

**Overall Accuracy:** 0.9450    
**Weighted Avg (P/R/F1):** 0.94 / 0.94 / 0.94

**Confusion Matrix (Actual × Predicted)**

| Actual \\ Predicted | Non-Flowering (0) | Flowering (1) |
| :--- | :--- | :--- |
| **Non-Flowering (0)** | 353 | 2 |
| **Flowering (1)** | 20 | 25 |

---

*Note: Decision thresholds used were 0.3600 for the Flower Classifier and 0.3850 for the Silique Classifier.*

### Summary of Tradeoffs

The final pipeline prioritizes precision, interpretability, and biological consistency over maximizing raw classification performance.

The petal classifier achieved strong performance and was highly reliable
The silique classifier was the primary bottleneck, with false positives reducing downstream flowering predictions
The strict flowering definition and conservative thresholds resulted in:
- High precision
- Moderate recall for flowering (~56%)

As a result, predicted flowering occurrences should be interpreted as a lower bound on true flowering activity, with a bias towards plants being in the early reproductive stage. These tradeoffs align with the goal of extracting reliable phenological signals from noisy, real-world image data.


## Inference

### Inference Pipeline
All images were transformed into embeddings using DINOv2 and then ran through the two trained classifiers. For each image probabilities for petal presence and silique presence were computed and thresholded. These predictions were then combined using the flowering rule defined above.

All image-level predictions, including probabilities and intermediate classifier outputs, were logged to enable traceability and downstream analysis.


### Occurrence-Level Aggregation

Final flowering labels were assigned at the occurrence level, rather than the image level.

An occurrence was classified as flowering if at least one associated image satisfied the flowering condition:

Occurrence flowering = At least 1 image flowering AND no siliques present in any image

This presence-based aggregation was chosen to reflect how many occurrence images were structured. Since many images for a single occurrence only captured parts of the plant at a time, the rule was chosen to make sure that the plant did indeed enter the reproductive stage, but had entered post-flowering yet due to the presence of any siliques.

### Inference Results
