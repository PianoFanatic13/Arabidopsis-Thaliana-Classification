# Arabidopsis Thaliana Flowering Classification using DINOv2

## Overview
This repository contains the models and analysis used to classify flowering stages of the *Arabidopsis thaliana* from field images using self-supervised visual representations derived from Meta's DINOv2 model. The project focuses on distinguishing flowering and non-flowering states, including the presence of siliques, to support phenological analysis and downstream biological research. This project was done under the guidance of WSU-USDA Researcher Dr. Xianran Li

The work emphasizes robust feature extraction from limited labeled data and evaluates the effectiveness of modern self-supervised vision transformers for fine-grained plant phenology classification.

## Research Motivation
The *Arabidopsis Thaliana L. Heynh* is one of the most widely studied organisms in plant biology. It serves as a model species, allowing insights gained from studying that plant to be directly applied to research of other more complex or less studied plants. And the flowering time is a key trait for understanding plant development and environmental response. While flowering phenology has been extensively studied in controlled laboratory and common garden experiments, much less is known about flowering patterns across large geographic areas, in the wild. 

At the same time, biodiversity platforms such as GBIF, which compile millions of georeferenced plant observations and images collected, many of which are contributed through citizen science efforts, like the iNaturalist platform. These images provide the opportunity to study phenology outside of controlled environments, but are also collected under variable conditions and without too much standardization for imaging protocols. As a result, these images introduce challenges like inconsistency in image quality, partial and obfuscated views of plants, and ambiguity between closely related reproductive stages, like early flowering and post-flowering. This variability makes phenological inference from images alone difficult, and motivates the structure-aware approach taken in this project.

This project explores whether modern computer vision techniques can be used to infer flowering stages of the *Arabidopsis thaliana* from unconstrained field images, at scale. Manually labeling flowering stages for a large collection of images is time-consuming and does not scale well, especially for phenological analysis at the continental-scale, which this project aims to develop. To address this bottleneck, we rely on self-supervised vision models that leverage visual structure learned from large amounts of unlabeled image data, reducing the need for extensive and tedious manual annotation of images. Inspired by recent work such as the FLORIST AI, this approach aims to extract biologically meaningful signals like flowering status from noisy, real-world images in a way that's reliable and robust. 


## Datasets

### Data Overview

The primary data for this project comes from the [GBIF](https://www.gbif.org/), which aggregates millions of georeferenced plant observations and images, many contributed through citizen science initiatives like iNaturalist. Each observation in GBIF is assigned a unique `gbifID`, which corresponds to a single occurrence of a plant species at a given time and location. An occurrence may have one or more associated images, often capturing multiple perspectives of the same plant or population. Of the occurrences that did have multiple images, most had around 3-4 images.

**Across the dataset:**
- Number of occurrences: 13,255
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
Because multiple images may correspond to the same GBIF occurrence, all modeling splits are performed in a **group-aware** manner at the occurrence level. This ensures that no images from the same observation appear across both training and evaluation sets, preventing data leakage and artificially inflated performance.


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

The final pipeline prioritizes precision, interpretability, and biological consistency over maximizing raw classification performance. Minimizing false positives through precision was one of the main considerations since introducing incorrect classifications would affect the subsequent analysis of inference results, and would thus require lots of manual inspection of the results.

- The petal classifier achieved strong performance and was highly reliable
- The silique classifier was the primary bottleneck, with false positives reducing downstream flowering predictions
- The strict flowering definition and conservative thresholds resulted in:
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

**Dataset-wide inference summary:**
- Total images processed: 22,398  
- Total occurrences evaluated: 13,255  

Following image-level classification and occurrence-level aggregation:

- Number of flowering occurrences: 694  
- Number of non-flowering occurrences: 12,561 


Due to the conservative flowering definition and thresholding strategy, predicted flowering occurrences represent a **high-confidence subset** of true flowering events. Given the observed recall (~56%), the model likely underestimates total flowering frequency, particularly in cases where siliques are incorrectly detected or petals are partially occluded.

These predictions allow downstream phenological analysis, including seasonal distributions and geographic trends.



## Phenological Analysis

Using the occurrence-level flowering predictions, we analyze large-scale flowering patterns of *Arabidopsis thaliana* across Europe.

### Seasonal Trends

Flowering occurrences were aggregated by month and day-of-year to examine seasonal patterns.

![Seasonal distribution of flowering and non-flowering occurrences](/analysis_images/seasonal_distribution.png "Optional Title Text")

Key observations:
- Flowering events show a clear distribution peaking during the Spring, primarily March and April
- The distribution aligns broadly with known phenological patterns from controlled studies

A regression line was also calculated to see if there was any correlation between higher latitudes in coordinates, and flowering time


![Regression line of flowering time vs latitude](/analysis_images/regression.png "Optional Title Text")
We can clearly see a trend of flowering occurring later at higher latitudes and it being significant with a p-value of '1.01e-05'.


### Geographic Trends

Flowering occurrence data were analyzed across geographic regions and latitudinal gradients.

![Flowering occurrences mapped over Europe](/analysis_images/occurrences_map.png "Optional Title Text")

![Flowering in different latitude bands](/analysis_images/latitude_bands.png "Optional Title Text")

| Month | South (<45°N) | Mid (45-55°N) | North (>55°N) |
| :--- | :---: | :---: | :---: |
| Jan | 16.7% (n=1) | 5.8% (n=14) | 0.0% (n=0) |
| Feb | 8.3% (n=3) | 7.8% (n=28) | 0.0% (n=0) |
| Mar | 6.8% (n=10) | 15.5% (n=304) | 18.5% (n=5) |
| Apr | 0.8% (n=1) | 4.0% (n=226) | 9.7% (n=31) |
| May | 1.4% (n=1) | 0.8% (n=19) | 2.4% (n=19) |
| Jun | 0.0% (n=0) | 0.8% (n=2) | 0.0% (n=0) |
| Jul | 0.0% (n=0) | 3.4% (n=3) | 2.4% (n=1) |
| Aug | 0.0% (n=0) | 3.1% (n=3) | 9.5% (n=2) |
| Sep | 0.0% (n=0) | 3.8% (n=4) | 0.0% (n=0) |
| Oct | 0.0% (n=0) | 2.4% (n=2) | 0.0% (n=0) |
| Nov | 0.0% (n=0) | 6.8% (n=5) | 0.0% (n=0) |
| Dec | 33.3% (n=1) | 7.6% (n=9) | 0.0% (n=0) |

*Note: `%` is the flowering rate based on all occurrences in the month, for each latitude; `n` is flowering occurrences. Values with very small `n` can look inflated (for example South in Dec).*

| Latitude Band | Overall Flowering Rate | Flowering Occurrences | Total Observations |
| :--- | :---: | :---: | :---: |
| Mid (45-55°N) | 5.4% | 619 | 11,515 |
| North (>55°N) | 4.3% | 58 | 1,344 |
| South (<45°N) | 4.3% | 17 | 395 |

Key observations:
- The middle latitude band shows signs of a strong unimodal distribution, showing flowering mainly in March and April  
- Warmer climates show earlier onset of flowering  
- There isn't enough data of flowering occurrences in the north and south to make any strong claims regarding their flowering distributions

### Interpretation

While the model does not capture all flowering events (due to silique-related errors and thresholds prioritizing precision), the detected patterns represent **high-confidence signals** of flowering activity in the Spring, with most occurrences being between the latitude of 45-55 within Europe, based on citizen-science data.

These results demonstrate that:
- Large-scale phenological trends can be extracted from noisy, real-world image data  
- Self-supervised visual models can support analysis without extensive manual labeling  


## Reproducibility

This repository is structured to support reproducibility of the modeling pipeline and analysis.

### Environment
* **Python:** 3.10.18
* **PyTorch:** 2.2.2
* **CUDA:** 11.8
* **Transformers:** ≥ 4.53.1
* **GPU Tested:** NVIDIA RTX 3060 Laptop GPU

### Installation
```bash
conda create -n dinov2 python=3.10.8
conda activate dinov2
pip install -r requirements.txt
```


## Limitations and Future Work

### Limitations
* **Silique detection errors:** False positives in the silique classifier reduced flowering recall and are the primary bottleneck for accuracy.
* **Conservative flowering definition:** The strict rule prioritizes early flowering stages and excludes ambiguous transitional states like going from first onsight of flowering to post-flowering.
* **Image variability:** Occluded images, low resolution, and inconsistent viewpoints limited detection performance, although the pipeline accounted for and adapted to this well.
* **Sampling bias:** GBIF data are not uniformly distributed across space or time.

### Future Work
* **Improved detection:** Improve silique detection through exploration of segmentation-based approaches or CNNs.
* **Phenological classification:** Explore multi-label phenological classification.
* **Temporal metadata:** Incorporate temporal metadata for longitudinal analysis.
* **Ecosystem expansion:** Expand to additional plant species and ecosystems.

## Acknowledgements

This project was conducted under the guidance of Dr. Xianran Li (WSU-USDA).

DINOv2 model provided by Meta AI Research

Data was provided by the Global Biodiversity Information Facility (GBIF):

> GBIF.org (14 August 2025) GBIF Occurrence Download  
> https://doi.org/10.15468/dl.2w3rzw

We also acknowledge the contributions of citizen scientists and platforms such as iNaturalist, whose observations made this work possible.
