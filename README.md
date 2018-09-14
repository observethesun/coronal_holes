[![Python](https://img.shields.io/badge/python->=3.5-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.4-orange.svg)](https://tensorflow.org)

## Segmentation of coronal holes with a convolutional neural network

The repository contains a source code that allows to reproduce the coronal holes segmentation model described in a paper [Segmentation of coronal holes in solar disk images with a convolutional neural network]() (submitted to MNRAS).

### Demo

Try a [demo](https://illarionovea.github.io/) running directly in the browser
to see how the model will process solar disk images you will feed to it.

### Installation and quickstart

Clone the repo
```
git clone https://github.com/observethesun/coronal_holes.git
```
or download it as an archive.

For quickstart you need to have an archive of SDO/AIA 193 A jpeg images in 1K resolution that 
correspond to annotated coronal holes in date and time of observation and check that 
all required libraries are installed. 

### Content

The repository is organized as follows:

* ```src``` folder contains the neural network architecture, methods for image loading and processing and classes that handle large datasets.
* ```notebooks``` folder contains tutorials that explain how everything works. More detailed, you may need
    * ```train_segmentation_model.ipynb``` to reproduce the model training procedure
    * ```run_segmentation_model.ipynb``` to test the model on SDO/AIA images
    * ```convert_images.ipynb``` to convert ```.jpeg``` images to more memory efficient ```.blosc``` file format.
* ```data``` folder provides a daily archive of coronal holes (CHs) in ```.abp``` file 
format. CHs were obtained from SDO/AIA 193 A images in 1K resolution for the period 2010 to 2018.
These ```.abp``` files are targets for the segmentation model.
* ```requirements.txt```  contains a list of python libraries required to run the code.
    
### What is an ```.abp``` file format

```.abp``` is a text format used at the [Kislovodsk Mountain Astronomical Station](http://en.solarstation.ru/) to describe active regions (in particular, coronal holes) isolated in solar disk images.

All ```.abp``` files have similar structure:
* Filename contains date and time of the solar disk observation. 
* First line of a file contains 7 numbers. First three numbers are x- and y-coordinates of the solar disk center and its radius given in pixel units. Last numbers are not important for CHs.
* Second line is not important for CHs.
* Third line contains a number of active regions.
* The rest lines are given by pairs of lines one for each active region. Pairs are organized as follows:
    * first line contains 7 numbers, where the first one is an index of active region within the current file, second one in a number of pixels occupied by active region. Following numbers are not important for CHs.
    * second line contains triples x, y, c one for each pixel within the active region. Here x, y are coordinates of pixel belonging to active region and c is a flag whether this pixel is inner (c=2) or edge (c=1) for the active region. Coordinates are in pixel units.


### Where to get SDO/AIA images

We suggest [http://jsoc.stanford.edu/](http://jsoc.stanford.edu/) for massive queries. For single images with preview consider [http://suntoday.lmsal.com/suntoday/](http://suntoday.lmsal.com/suntoday/).

### Why ```.blosc``` not ```.jpeg```

During the neural network training we read images from disk multiple times. However, I/O operation are known to be slow. To speed up the process we suggest to keep data in optimized formats that can be operated faster.  Using ```.blosc``` format one can have a speed benefit up to several times. Read more at [http://blosc.org/](http://blosc.org/).


### Citing this work

Using provided ```.abp``` files in your research please cite 

```
Tlatov, A., Tavastsherna, K. & Vasil’eva, V. Sol Phys (2014) 289: 1349.
```

[![DOI](https://zenodo.org/badge/DOI/10.1007/s11207-013-0387-4.svg)](https://doi.org/10.1007/s11207-013-0387-4)
