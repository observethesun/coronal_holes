[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.4-orange.svg)](https://tensorflow.org)

## Segmentation of coronal holes with a convolutional neural network

This repository contains the source code to reproduce the coronal holes segmentation model from the paper [Segmentation of coronal holes in solar disk images with a convolutional neural network](https://arxiv.org/abs/1809.05748) (published in [MNRAS](https://doi.org/10.1093/mnras/sty2628)).

Note that the code was updated to be compatible with [helio](https://github.com/observethesun/helio) framework.
Checkout to the branch [mnras2018](https://github.com/observethesun/coronal_holes/tree/mnras2018) for original version.

### Demo

Try a [demo](https://illarionovea.github.io/) running directly in the browser
to see how the model will process solar disk images you will feed to it.
Note that the model was optimized to SDO/AIA images obtained from [SunInTime](https://suntoday.lmsal.com/suntoday/)
website. 

### Installation

Clone the repo
```
git clone --recursive https://github.com/observethesun/coronal_holes.git
```

### Usage

The code is based on [helio](https://github.com/observethesun/helio) framework. See the API [documentation](http://observethesun.github.io/helio/) to learn more about its features.

A dataset proposed for model training consists of SDO/AIA 193 Angstrom solar disk images in 1K
resolution obtained from [SunInTime](https://suntoday.lmsal.com/suntoday/) website and a dataset
of coronal holes regions provided by the
[Kislovodsk Mountain Astronomical Station](http://en.solarstation.ru/).

The notebook [Train_segmentation_model](./notebooks/1.Train_segmentation_model.ipynb) 
contains data preprocessing, neural network architecture and model training pipeline.
The notebook [Apply_segmentation_model](./notebooks/2.Apply_segmentation_model.ipynb) 
demonstrates inference pipeline.

### Citing this work

```
Illarionov E., Tlatov А., 2018, MNRAS, 481, 4.
```

[![DOI](https://zenodo.org/badge/DOI/10.1093/mnras/sty2628.svg)](https://doi.org/10.1093/mnras/sty2628)
