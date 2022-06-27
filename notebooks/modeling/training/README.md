# Modeling - Image Super Resolution

The focus of these notebooks is to leverage the ( Low Resolution (LR) - High Resolution (HR) ) satellite image pairs downloaded and processed from [Google Earth Engine](https://earthengine.google.com/) APIs using state-of-the-art (SOTA) image super resolution deep learning models. 

<br>

## Notebooks for Modeling Pipelines

There are a few folders which consist of necessary code to implement, train and test the above super resolution deep learning models.

<br>

### training

Consists of notebooks to train ISR models including RDN, RRDN, EDSR, EDSR + SRGAN

1.  [ISR__training_helper.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/training/ISR__training_helper.ipynb) - The purpose of the notebook is to automatically clean up the Google Drive Trash (permanently delete contents) to prevent space shortage during training, as well as to rename the logs/weight files from datetime to hyperparameters tested, to facilitate viewing in TensorBoard.

<br>

2.  [ISR_training_RDN_x3_PSNR.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/training/ISR_training_RDN_x3_PSNR.ipynb) - The purpose of this notebook is to set up a RDN model from the ISR framework and train it.

<br>

3.  [ISR_training_RRDN.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/training/ISR_training_RRDN.ipynb) - The purpose of this notebook is to set up a RRDN model from the ISR framework and train it.

<br>

4.  [SRGAN_Training_L8_S2.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/training/SRGAN_Training_L8_S2.ipynb) - This notebook is used to train the EDSR model. We also save the EDSR pre-trained model and then couple it with the SRGAN architecture to fine-tune it.

<br>
