
# Satellite Image Super Resolution Notebooks

This folder contains all the relevant notebooks covering the end-to-end workflow to build a Super Resolution Imaging System leveraging Deep Transfer Learning and Computer Vision for Satellite Imagery from Google Earth Engine. There are primarily three major steps and each have dedicated folders.

 -  [Data Sourcing and Processing](./data_sourcing_and_processing) 
 -  [Modeling](./modeling) 
 -  [Model Evaluation](./evaluation) 
 
<br>

## Data Sourcing and Processing

The focus of this step is to leverage [Google Earth Engine](https://earthengine.google.com/) APIs to retrieve satellite imagery data from majorly two satellites:

- [Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR)
- [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR)

Google Earth Engine combines a multi-petabyte catalog of satellite imagery and geospatial datasets with planetary-scale analysis capabilities. Scientists, researchers, and developers use Earth Engine to detect changes, map trends, and quantify differences on the Earth's surface. Earth Engine is now available for commercial use, and remains free for academic and research use.

| ![](https://i.imgur.com/Bd5Y7ze.png) | 
|:--:| 
| *Source: https://earthengine.google.com* |


__Note:__ Remember the focus here is to leverage Landsat data as inputs which are satellite images of low resolution and Sentinel-2 data as output (gold standard) references, which are satellite images of high resolution.

<br>

## Modeling - Image Super Resolution

The focus of these notebooks is to leverage the ( Low Resolution (LR) - High Resolution (HR) ) satellite image pairs downloaded and processed from [Google Earth Engine](https://earthengine.google.com/) APIs using state-of-the-art (SOTA) image super resolution deep learning models. 

<br>

The general workflow here is to train a deep learning model to learn relevant relationships between the LR-HR images pairs, based on extracted features using models like Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs). Once the model is trained, in the future, if a LR image is given to the model, it can predict a SR (super-resolution) image which is almost equivalent of a HR (high-resolution) gold standard reference image.

<br>

| ![](https://i.imgur.com/JutPDZA.png) | 
|:--:| 
| *Source: Created by Authors* |

<br>

## Model Evaluation

The focus of these notebooks is to leverage various mathematical metrics to quantify the quality of the generated super resolution images and compare it to the gold standard reference high resolution images. Leverages open-source frameworks like [cpbd](https://pypi.org/project/cpbd/), [sewar](https://github.com/andrewekhalel/sewar) and [up42](https://github.com/up42/image-similarity-measures) to assess similarity between ( generated super-resolution (SR) image - reference high-resolution (HR) ) image pairs.

The following visual shows a comparative analysis of the [Cumulative Probability of Blur Detection (CPBD)](https://ivulab.asu.edu/software/cpbd/) metric across multiple super resolution models on our satellite data.

| ![](https://i.imgur.com/CiOw1CN.png) | 
|:--:| 
| *Source: Created by Authors* |
