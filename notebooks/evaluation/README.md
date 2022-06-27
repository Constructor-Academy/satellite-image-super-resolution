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

1.  [compare_image_sharpness.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/evaluation/compare_image_sharpness.ipynb) - This notebook leverages the [Cumulative Probability of Blur Detection (CPBD)](https://ivulab.asu.edu/software/cpbd/) metric to compare between predicted and actual high resolution images. This is a perceptual-based no-reference objective image sharpness metric based on the cumulative probability of blur detection developed at the [Image, Video and Usability Laboratory of Arizona State University](https://ivulab.asu.edu/Quality/CPBD).
> This metric is based on the study of human blur perception for varying contrast values. The metric utilizes a probabilistic model to estimate the probability of detecting blur at each edge in the image, and then the information is pooled by computing the cumulative probability of blur detection (CPBD). Higher CPBD values represent sharper images. Images having small CPBD value denote blurred and noisy images.

<br>

2.  [get_image_similarity_sewar.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/evaluation/get_image_similarity_sewar.ipynb) - This notebook gets image similarity metrics between reference HR images (sentinel) and SR images (model predictions). It uses the [sewar](https://github.com/andrewekhalel/sewar) open-source framework. 

<br>

3.  [get_image_similarity_up42.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/evaluation/get_image_similarity_up42.ipynb) - This notebook gets image similarity metrics between reference HR images (sentinel) and SR images (model predictions). It uses the [op42](https://github.com/up42/image-similarity-measures) open-source framework.

<br>

4.  [model_image_comparision.ipynb](https://colab.research.google.com/github/dipanjanS/satellite-image-super-resolution/blob/master/notebooks/evaluation/model_image_comparision.ipynb) - The purpose of this notebook is to visualize model predictions for a sample image side-by-side, comparing to high-resolution ground truth and low-resolution baseline or model predictions manually.

<br>

### Sample Results

The following visual shows a comparative analysis of the [Cumulative Probability of Blur Detection (CPBD)](https://ivulab.asu.edu/software/cpbd/) metric across multiple super resolution models on our satellite data.

| ![](https://i.imgur.com/CiOw1CN.png) | 
|:--:| 
| *Source: Created by Authors* |

The EDSR + SRGAN fine-tuned models definitely show more promise than the other models!
