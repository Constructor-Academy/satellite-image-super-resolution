1. tif_exporter_moving_square - connect to google earth engine and scan a geographic area \ region by setting latitude and longitude
can scan across a given area, shifting by 80 km2 diameter and go for a certain number of steps and source images from a particular area (mention areas)

2. tif_Filtering_post_download - check pixel values in terms of percentiles for cloud filtering, 
using cloud scores and not downloading images having cloud score above a threshold
some images have blackouts so using lowest percentile (10%ile) and take out bad images
filtering also takes into account images from landsat and sentinel is captured within 7 days
to get as clean images as possible
basically make sure good images are going into training

3. tif manual filtering - manual filtering of images

4. tif_pairwise_plotter - to see images post download and visualize them

5. final_dataset_rename_files - to rename files to have same name for LR and HR pairs before training the model (gives a unique index based on latitude, longitude, year and month)

6. train val split - split data into train \ val splits

7. GDAL_transformer_manual_stretch - scale both landsat and sentinel images as they are very different ranges, scale them into 8 bit range, adjust brightness and color temparature values
also resizing the images 1:3 LR:HR image sizes and also doing cropping to remove some minor pixel boundaries to enable the 1:3 

8. GDAL transformer PNG