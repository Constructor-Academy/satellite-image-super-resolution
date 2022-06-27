# Model Evaluation

The focus of these notebooks is to leverage various mathematical metrics to quantify the quality of the generated super resolution images and compare it to the gold standard reference high resolution images.

| ![](https://i.imgur.com/Iq9EwDt.png) | 
|:--:| 
| *Source: https://blog.eyewire.org/behind-the-science-about-the-data/* |

Leverages open-source frameworks like [cpbd](https://pypi.org/project/cpbd/), [sewar](https://github.com/andrewekhalel/sewar) and [up42](https://github.com/up42/image-similarity-measures) to assess similarity between ( generated super-resolution (SR) image - reference high-resolution (HR) ) image pairs. There are a variety of metrics available which include:

- [Cumulative Probability of Blur Detection (CPBD)](https://ivulab.asu.edu/software/cpbd/)
- [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) 
- [Root Mean Sqaured Error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
- [Peak Signal-to-Noise Ratio (PSNR)](https://ieeexplore.ieee.org/abstract/document/1284395/)
- [Structural Similarity Index (SSIM)](https://ieeexplore.ieee.org/abstract/document/1284395/)
- [Universal Quality Image Index (UQI)](https://ieeexplore.ieee.org/document/995823/)
- [Multi-scale Structural Similarity Index (MS-SSIM)](https://ieeexplore.ieee.org/abstract/document/1292216/)
- [Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)](https://hal.archives-ouvertes.fr/hal-00395027/)
- [Spatial Correlation Coefficient (SCC)](https://www.tandfonline.com/doi/abs/10.1080/014311698215973)
- [Relative Average Spectral Error (RASE)](https://ieeexplore.ieee.org/document/1304896/)
- [Spectral Angle Mapper (SAM)](https://ntrs.nasa.gov/search.jsp?R=19940012238)
- [Spectral Distortion Index (D_lambda)](https://www.ingentaconnect.com/content/asprs/pers/2008/00000074/00000002/art00003)
- [Spatial Distortion Index (D_S)](https://www.ingentaconnect.com/content/asprs/pers/2008/00000074/00000002/art00003)
- [Quality with No Reference (QNR)](https://www.ingentaconnect.com/content/asprs/pers/2008/00000074/00000002/art00003)
- [Visual Information Fidelity (VIF)](https://ieeexplore.ieee.org/abstract/document/1576816/)
- [Block Sensitive - Peak Signal-to-Noise Ratio (PSNR-B)](https://ieeexplore.ieee.org/abstract/document/5535179/)
- [Feature-based similarity index (FSIM)](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf)
- [Information theoretic-based Statistic Similarity Measure (ISSM)](https://www.tandfonline.com/doi/full/10.1080/22797254.2019.1628617)
- [Signal to reconstruction error ratio (SRE)](https://www.sciencedirect.com/science/article/abs/pii/S0924271618302636)

and many more! The idea here is to evaluate our predictions against the actuals by using some of these metrics.

<br>

### Notebooks for Model Evaluation

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

8. [GDAL_transformer_PNG.ipynb<sup>*</sup>](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/GDAL_transformer_PNG.ipynb) - This notebook is used for converting TIF files sourced from Google Earth Engine into PNG files, and processed with [GDAL_transformer_manual_strech.ipynb](https://nbviewer.org/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/data_sourcing_and_processing/GDAL_transformer_manual_strech.ipynb), with resizing to ensure same image dimensions.


<br>

_<b>* Large file alert!</b> Please use [NBViewer](https://nbviewer.org/) to view these notebooks instead of direct GitHub. Links already created and embedded here for convenience._
