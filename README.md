# Arabidopsis Thaliana Flowering Classification using DINOv2

## Overview
This repository contains the models and analysis used to classify flowering stages of the *Arabidopsis thaliana* from field images using self-supervised visual representations derived from Meta's DINOv2 model. The project focuses on distinguishing flowering and non-flowering states, including the presence of siliques, to support phenological analysis and downstream biological research. This project was done under the supervision of USDA Researcher Dr. Xianran Li

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
In this project, we use a very strict definition for what was considered flowering for a plant. Plants were classified as flowering if they were in the early stages of flowering, right from the time the first anthers were visible, to when it starts reproducing and siliques, or seed pods, were visible, a stage pants would only be in for a couple weeks. *Find out specific flowering timeline how long its in that state.* From this we can derive simple visual cues as to help classify flowering stage, if a plant has flower petal and no siliques, then it's flowering, whereas if it has siliques at all, then it's in the post-flowering stage and not flowering. To help spot these two visual traits in plant images, the pipeline uses two independent binary classifiers for images, one for petal presence, and the other for silique presence. *Maybe include the bold part for the flowering rule

### Pipeline
A small set of images labeled for petal and silique presence were used for training and testing. To prevent data leaks, dataset splits are group-aware in regards to Occurrence ID. Since each occurrence may contain multiple images, during training and testing, only one image per occurrence was included in each split, ensuring similar images of the same plant don't appear across both the train and test sets. The training sets for both the silique and petal classifiers were sampled with balanced classes to address the issue of class imbalance across the overall image data. The testing set was sampled randomly and is imbalanced to allow us to get a better sense of how the models would perform on the imbalanced dataset for inference. 

### Model
* **Backbone:** `facebook/dinov3-vitl16-pretrain-lvd1689m`
* **Pretraining:** Self-supervised learning on large-scale natural image datasets
* **Embedding Strategy:**
    * Global pooled representation from the final hidden state
    * Frozen backbone during downstream training

### Classification Pipeline
1.  Image preprocessing using the DINOv3 image processor.
2.  Feature extraction via frozen DINOv3 backbone.
3.  Lightweight classifier trained on top of embeddings.
4.  Threshold tuning and evaluation on a held-out test set.

## Results

### Performance Summary

| Task | Metric | Value |
| :--- | :--- | :--- |
| **Flower Classification** | Accuracy | XX% |
| **Flower Classification** | F1-score | XX |
| **Silique Classification** | Accuracy | XX% |

### Observations
* DINOv2 embeddings capture subtle morphological cues relevant to flowering stages.
* Most classification errors occur in transitional growth stages (e.g., senescing flowers developing into siliques).
* Detailed metrics, confusion matrices, and plots are available in the `results/` directory.

## Reproducibility

### Environment
* **Python:** 3.10+
* **PyTorch:** 2.5.1
* **CUDA:** 11.8
* **Transformers:** ≥ 4.48
* **GPU Tested:** NVIDIA RTX 3060 Laptop GPU

Installation
conda create -n dinov2 python=3.10
conda activate dinov3
pip install -r requirements.txt



## Limitations

Limited labeled data for certain transitional growth stages

Model performance may degrade on extreme lighting or occlusion

Dataset primarily reflects images from specific geographic regions


Acknowledgements
