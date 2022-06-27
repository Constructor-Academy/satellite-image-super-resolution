# Modeling - Image Super Resolution

The focus of these notebooks is to leverage the ( Low Resolution (LR) - High Resolution (HR) ) satellite image pairs downloaded and processed from [Google Earth Engine](https://earthengine.google.com/) APIs using state-of-the-art (SOTA) image super resolution deep learning models. 

<br>

## Image Super Resolution Workflow

The general workflow here is to train a deep learning model to learn relevant relationships between the LR-HR images pairs, based on extracted features using models like Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs). Once the model is trained, in the future, if a LR image is given to the model, it can predict a SR (super-resolution) image which is almost equivalent of a HR (high-resolution) gold standard reference image.

<br>

<a target="_blank" href="#">
  <img src="https://i.imgur.com/JutPDZA.png" align="center"/>
</a> 

<br>

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

<br>

### Residual in Residual Dense Network (RRDN) with GANs

<img src="https://i.imgur.com/C25GtNh.png" align="center"/>

<br>
Residual in Dense Networks (RRDN) is similar to SRResNet Model which is actually used in the [SRGAN paper](https://arxiv.org/abs/1609.04802) but instead here, the basic blocks will be replaced with RRDBs - Residual in Residual Dense Blocks. The RRDBs combines multi-level residual network and dense connections. Removing BN layers has proven to increase performance. Also it helps remove unpleasant artifacts and improve generalization ability. It uses a relativisting discriminator when training with the GAN (Generative Adversarial Network), which helps preidct the probability that a real image is relatively more realistic than a fake one.

<br>

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

The EDSR model follows this high-level architecture (Figure 3 in mentioned paper as depicted below) is described in the paper [Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)](https://arxiv.org/abs/1707.02921). It is a winner of the [NTIRE 2017 super-resolution challenge](http://www.vision.ee.ethz.ch/ntire17/). 

<img src="https://i.imgur.com/xQTPeBh.png" align="center"/>

The significant performance improvement in this model is due to optimization by removing unnecessary modules in conventional residual networks. Specifically, they improved the performance by employing a better ResNet structure.









<br>

### Notebooks for Data Sourcing and Processing Pipelines

<br>

1.  [tif_exporter_moving_square.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_exporter_moving_square.ipynb) - The notebook has the necessary workflow to enable the following:
>  - connecting to Google Earth Engine and scanning a selected geographical area \ region by setting the relevant latitude and longitude
>  - scan across a given area, shifting by 80 km<sup>2</sup> diameter downloading images from two satellite sources (landsat-8, low-resolution (LR) and sentinel-2, high resolution (HR) ) to Google Drive
>  - images with clouds and no-data pixels are discarded, and a temporal matching is done, so that only one image pair is downloaded per month, with a restriction that the images are within `max_days_apart`

<br>

2.  [tif_filtering_post_download.ipynb<sup>*</sup>](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_filtering_post_download.ipynb) - This notebook is used for secondary filtering of images sourced by [tif_exporter_moving_square.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_exporter_moving_square.ipynb), to remove HR-LR image pairs where at least one of the images has cloud content or missing data. The idea is to make sure good images are going into training. Key features include:
>  - check pixel values in terms of percentiles for cloud filtering, using cloud scores and not downloading images having cloud score above a threshold
>  - some images have blackouts so use lowest percentile (10%ile) to take out bad images
>  - filtering also takes into account images from landsat and sentinel is captured within 7 (or `max_days_apart`) days to get as clean images as possible

<br>

3.  [tif_manual_filtering.ipynb<sup>*</sup>](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_manual_filtering.ipynb) - This notebook is used for final filtering of images sourced by [tif_exporter_moving_square.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_exporter_moving_square.ipynb), and filtered with [tif_filtering_post_download.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_filtering_post_download.ipynb), to visually check and remove HR-LR image pairs where at least one of the images has cloud content or missing data.

<br>

4.  [tif_pairwise_plotter.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_pairwise_plotter.ipynb) - This notebook is for plotting of high-resolution sentinel and low-resolution landsat image pairs that have been downloaded from Google Earth Engine as TIF files. Different scaling is applied to sentinel and landsat pixel values to equalize image brightness and adjust for the different data ranges of the original files. For non-filtered data, cloud and missing-data pixel filters can be applied to only plot high-quality images.

<br>

5.  [final_dataset_rename_files.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/final_dataset_rename_files.ipynb) - This notebook is used for renaming the image files sourced with [tif_exporter_moving_square.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/tif_exporter_moving_square.ipynb), so that each HR-LR image pair has a unique index.

<br>

6.  [train_val_split.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/train_val_split.ipynb) - This notebook is used for splitting the sourced image files into training and validation datasets.

<br>

7. [GDAL_transformer_manual_strech.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/GDAL_transformer_manual_strech.ipynb) - This notebook is used to resize and scale the landsat and sentinel TIF images downloaded from Google Earth Engine, plotting the results. The output is to be used by [GDAL_transformer_PNG.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/GDAL_transformer_PNG.ipynb). Key features include:
>  - scale both landsat and sentinel images as they are very different ranges, scale them into 8 bit range
>  - adjust brightness and color temparature values
>  - resizing the images 1:3 LR:HR image sizes
>  - cropping to remove some minor pixel boundaries to enable the 1:3 ratio

<br>

8. [GDAL_transformer_PNG.ipynb<sup>*</sup>](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/GDAL_transformer_PNG.ipynb) - This notebook is used for converting TIF files sourced from Google Earth Engine into PNG files, and processed with [GDAL_transformer_manual_strech.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/GDAL_transformer_manual_strech.ipynb), with resizing to ensure same image dimensions.


<br>

_<b>* Large file alert!</b> Please use [NBViewer](https://nbviewer.org/) to view these notebooks instead of direct GitHub. Links already created and embedded here for convenience._
