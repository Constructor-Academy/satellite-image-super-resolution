# Modeling - Image Super Resolution

The focus of these notebooks is to leverage the ( Low Resolution (LR) - High Resolution (HR) ) satellite image pairs downloaded and processed from [Google Earth Engine](https://earthengine.google.com/) APIs using state-of-the-art (SOTA) image super resolution deep learning models. 

<br>


## Notebooks for Modeling Pipelines

There are a few folders which consist of necessary code to implement, train and test the above super resolution deep learning models.

<br>

### utils

Consists of general utility functions especially to handle modifications in existing libraries for training models like RDNs

1.  [ISR_module_adjustments.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/utils/ISR_module_adjustments.ipynb) - The purpose of this notebook is to adjust the functions of the original ISR framework to make them compatible for working with raw TIF files sourced from Google Earth Engine. If using PNG satellite images (pre-processed with `GDAL_transformer_PNG.ipynb`), use `ISR_module_adjustments_PNG.ipynb`.<br> 
Usage: the cells in this notebook should be copied to the training/prediction notebooks and ran at the beginning (if ISR is imported through `pip install`). Alternatively, these changes can be applied to a local ISR repo.

<br>

2.  [ISR_module_adjustments_PNG.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/utils/ISR_module_adjustments_PNG.ipynb) - The purpose of this notebook is to adjust the functions of the original ISR framework to make them compatible for working with PNG satellite files sourced from Google Earth Engine and pre-processed with `GDAL_transformer_PNG.ipynb`.<br>
Usage: the cells in this notebook should be copied to the training/prediction notebooks and ran at the beginning (if ISR is imported through `pip install`). Alternatively, these changes can be applied to a local ISR repo.

<br>
