# Data Sourcing and Processing

The focus of these notebooks is to leverage [Google Earth Engine](https://earthengine.google.com/) APIs to retrieve satellite imagery data from majorly two satellites:

- [Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR)
- [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR)

### Landsat 8

<a target="_blank" href="#">
  <img src="https://i.imgur.com/ojkQ4D5.png" width="130" align="left"/>
</a> 

This dataset is the atmospherically corrected surface reflectance from the Landsat 8 OLI/TIRS sensors. These images contain 5 visible and near-infrared (VNIR) bands and 2 short-wave infrared (SWIR) bands processed to orthorectified surface reflectance, and two thermal infrared (TIR) bands processed to orthorectified brightness temperature. These data have been atmospherically corrected using LaSRC and includes a cloud, shadow, water and snow mask produced using CFMASK, as well as a per-pixel saturation mask.


### Sentinel-2

<a target="_blank" href="#">
  <img src="https://i.imgur.com/Rvw7f51.png" width="130" align="left"/>
</a> 

Sentinel-2 is a wide-swath, high-resolution, multi-spectral imaging mission supporting Copernicus Land Monitoring studies, including the monitoring of vegetation, soil and water cover, as well as observation of inland waterways and coastal areas. The Sentinel-2 L2 data are downloaded from scihub. They were computed by running sen2cor. WARNING: ESA did not produce L2 data for all L1 assets, and earlier L2 coverage is not global. The assets contain 12 UINT16 spectral bands representing SR scaled by 10000 (unlike in L1 data, there is no B10). There are also several more L2-specific bands. In addition, three QA bands are present where one (QA60) is a bitmask band with cloud mask information.

Google Earth Engine combines a multi-petabyte catalog of satellite imagery and geospatial datasets with planetary-scale analysis capabilities. Scientists, researchers, and developers use Earth Engine to detect changes, map trends, and quantify differences on the Earth's surface. Earth Engine is now available for commercial use, and remains free for academic and research use.

| ![](https://i.imgur.com/Bd5Y7ze.png) | 
|:--:| 
| *Source: https://earthengine.google.com* |


__Note:__ Remember the focus here is to leverage Landsat data as inputs which are satellite images of low resolution and Sentinel-2 data as output (gold standard) references, which are satellite images of high resolution.

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
