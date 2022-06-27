# Modeling - Image Super Resolution

The focus of these notebooks is to leverage the ( Low Resolution (LR) - High Resolution (HR) ) satellite image pairs downloaded and processed from [Google Earth Engine](https://earthengine.google.com/) APIs using state-of-the-art (SOTA) image super resolution deep learning models. 

<br>

## Image Super Resolution Workflow

The general workflow here is to train a deep learning model to learn relevant relationships between the LR-HR images pairs, based on extracted features using models like Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs). Once the model is trained, in the future, if a LR image is given to the model, it can predict a SR (super-resolution) image which is almost equivalent of a HR (high-resolution) gold standard reference image.

<br>

<a target="_blank" href="#">
  <img src="https://i.imgur.com/JutPDZA.png" align="center"/>
</a> 

<br><br>

## Super Resolution Model Architectures

A fair amount of research was done to explore current state-of-the-art (SOTA) deep learning model architectures for image super resolution. Given the sensitive nature of the time at hand for the project and the focus to implement a super resolution system, the focus was given to SOTA model which also had an implementation framework or at least proof that these models could be implemented from available research using open-source deep learning libraries like TensorFlow or PyTorch. In the end we settled on the following model architectures:

The implemented networks include:

- Residual Dense Network (RDN) 
    - [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)
- Residual in Residual Dense Network (RRDN) with GANs
    - [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1609.04802)
- Enhanced Deep Residual Networks (EDSR) 
    - [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)
- Modified SRGAN (fine-tuning EDSR)
    - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

<br>

### Residual Dense Network (RDN)

<img src="https://i.imgur.com/JYLt1qM.png" align="center"/>

<br>
Residual Dense Networks (RDN) tries to address problems of not using hierarchical features from the original low-resolution (LR) images in image SR. It fully exploits
the hierarchical features from all the convolutional layers. Specifically, the paper proposes residual dense blocks (RDBs) to extract abundant local features via dense connected convolutional layers. RDBs further allows direct connections from the state of preceding RDB to all the layers of current RDB, leading to a contiguous memory (CM) mechanism. Local feature fusion in RDB is then used to adaptively learn more effective features from preceding and current local features and stabilizes the training of wider network. After fully obtaining dense local features, the model uses global feature fusion to jointly and adaptively learn global hierarchical features
in a holistic way. 

<br><br>

### Residual in Residual Dense Network (RRDN) with GANs

<img src="https://i.imgur.com/C25GtNh.png" align="center"/>

<br>
Residual in Dense Networks (RRDN) is similar to SRResNet Model which is actually used in the [SRGAN paper](https://arxiv.org/abs/1609.04802) but instead here, the basic blocks will be replaced with RRDBs - Residual in Residual Dense Blocks. The RRDBs combines multi-level residual network and dense connections. Removing BN layers has proven to increase performance. Also it helps remove unpleasant artifacts and improve generalization ability. It uses a relativisting discriminator when training with the GAN (Generative Adversarial Network), which helps preidct the probability that a real image is relatively more realistic than a fake one.

<br><br>

### Enhanced Deep Residual Networks (EDSR)

The EDSR model follows this high-level architecture (Figure 3 in mentioned paper as depicted below) is described in the paper [Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)](https://arxiv.org/abs/1707.02921). It is a winner of the [NTIRE 2017 super-resolution challenge](http://www.vision.ee.ethz.ch/ntire17/). 

<img src="https://i.imgur.com/xQTPeBh.png" align="center"/>

The significant performance improvement in this model is due to optimization by removing unnecessary modules in conventional residual networks. Specifically, they improved the performance by employing a better ResNet structure.

<img src="https://i.imgur.com/qdXew0b.png" align="center"/>

In Fig. 2 of the original paper as depicted above, they compared the building blocks of each network model from original ResNet, SRResNet, and the proposed network (EDSR). They removed the batch normalization layers from the network as Nah et al. presented in their image deblurring work. Since batch normalization layers
normalize the features, they get rid of range flexibility from networks by normalizing the features, it is better to remove them. They also experimentally show that this simple modification increases the performance substantially.

The performance is further improved by expanding the model size while they stabilized the training procedure. They also proposed a new multi-scale deep super-resolution system (MDSR) and training method, which can reconstruct high-resolution images of different upscaling factors in a single model as shown the Fig. 5 of the original paper as depicted below.

<img src="https://i.imgur.com/GrNRm6c.png" align="center"/>

They designed the baseline (multi-scale) models to have a single main branch with `B = 16` residual blocks so that most of the parameters are shared across different
scales as shown in Fig. 5. from the original paper above. In their multi-scale architecture, they introduced scale specific processing modules to handle the super-resolution at multiple scales. Their final MDSR has approximately 5 times more depth compared to the baseline multi-scale model, only 2.5 times more parameters are required, as the residual blocks are lighter than scale-specific parts.

<br>

### Modified SRGAN - Fine-tuning Enhanced Deep Residual Networks (EDSR)

The [SRGAN paper]() uses SRResNet as super-resolution model, a predecessor of EDSR. They proposed a generative adversarial network (GAN) for image super-resolution (SR). It was the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, they proposed a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes their solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, they use a content loss motivated by perceptual similarity instead of similarity in pixel space.

<img src="https://i.imgur.com/GnvUsGW.png" align="center"/>

The original SRGAN architecture is depicted in Fig. 4 from the original paper as shown above. However we do NOT use the generator SRResNet model depicted in the above figure. __Instead we leverage transfer learning principles where we use a pre-trained EDSR model (mentioned earlier) and fine-tune it using the SRGAN discriminator just like a normal GAN would be trained, except here the generator is a pre-trained EDSR model. This enables us to get better results than a vanilla SRGAN model.__

The SRGAN model uses a perceptual loss function composed of a content loss and an adversarial loss. The content loss compares deep features extracted from SR and HR images with a pre-trained [VGG network](https://arxiv.org/abs/1409.1556) _ϕ_.

![](https://i.imgur.com/odC8NBI.png)

where _ϕ<sub>l</sub>(I)_ is the feature map at layer _l_ and _H<sub>l</sub>_, _W<sub>l</sub>_ and _C<sub>l</sub>_ are the height, width and number of channels of that feature map, respectively. They also train their super-resolution model as generator _G_ in a generative adversarial network (GAN). The GAN discriminator _D_ is optimized for discriminating SR from HR images whereas the generator is optimized for generating more realistic SR images in order to fool the discriminator. They combine the generator loss depicted below

![](https://i.imgur.com/mvvw7u5.png)

with the content loss to a perceptual loss which is used as optimization target for super-resolution model training:

![](https://i.imgur.com/ENyrWln.png)

Instead of training the super-resolution model i.e. the generator from scratch in a GAN, they pre-train it with a pixel-wise loss and fine-tune the model with a perceptual loss. 

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

### prediction

Consists of a notebook to load a pre-trained super-resolution model and generate high resolution images from low resolution import images. Used for evaluation.

1.  [ISR_prediction.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/modeling/prediction/ISR_prediction.ipynb) - The purpose of this notebook is to create super-resolution images from a set of low-resolution images using a pre-trained ISR network.
