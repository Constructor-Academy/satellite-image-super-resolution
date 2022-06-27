# Modeling - Image Super Resolution

The focus of these notebooks is to leverage the ( Low Resolution (LR) - High Resolution (HR) ) satellite image pairs downloaded and processed from [Google Earth Engine](https://earthengine.google.com/) APIs using state-of-the-art (SOTA) image super resolution deep learning models. 

<br>

## Notebooks for Modeling Pipelines

There are a few folders which consist of necessary code to implement, train and test the above super resolution deep learning models.

<br>

### prediction

Consists of a notebook to load a pre-trained super-resolution model and generate high resolution images from low resolution import images. Used for evaluation.

1.  [ISR_prediction.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/prediction/ISR_prediction.ipynb) - The purpose of this notebook is to create super-resolution images from a set of low-resolution images using a pre-trained ISR network.
