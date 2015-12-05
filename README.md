# Part Constellation Models

This is the code used in our paper "Neural Activation Constellations: Unsupervised Part Model Discovery with Convolutional Networks" by Marcel Simon and Erik Rodner published at ICCV 2015. 
If you would like to refer to this work, please cite the corresponding paper

    @inproceedings{Simon15:NAC,
	author = {Marcel Simon and Erik Rodner},
	booktitle = {International Conference on Computer Vision (ICCV)},
	title = {Neural Activation Constellations: Unsupervised Part Model Discovery with Convolutional Networks},
	year = {2015},
    }

The following steps will guide you through the usage of the code.

## 1. Setup
1. Open Matlab and go to the folder containing this package
2. Run setup.m to download all libraries
3. Go to lib/caffe_pp and make it, you will need to create a Makefile.config. If you have an existing caffe, use that Makefile.config from there BUT DO NOT USE ANY EXISTING CAFFE as caffe_pp is a modified version.
4. Execute `make mat` in `lib/caffe_pp`
5. Go to lib/liblinear-2.1 and make it
6. Go to lib/liblinear-2.1/matlab and make it

## 2. Running the code

The `script.m` in the root folder of the package is all you need. You want to override the paths to the data set by passing them as name-value-pairs, for example `start('basedir','/path/to/dataset/')`. For more options, open it to see all options. Just pass additional parameters by adding name-value-pairs: `start('basedir','/path/to/dataset/','cnn_dir','./cnn_finetuning/vgg19/','crop_size',224);`.

The dataset files should contain a list of absolute image paths, a list of corresponding labels starting from 1, and a list of the corresponding assignment to train and test, where 1 indicates training and 0 test. 


imagelist.txt

```
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/image3.jpg
/path/to/image4.jpg
/path/to/image5.jpg
...
```

labels.txt

```
1
1
1
2
2
2
...
```

tr_ID.txt

```
0
1
1
0
1
1
...
```

## 3. Testing the models from the paper
The models of the paper are available at [https://drive.google.com/file/d/0B6VgjAr4t_oTQXN2Y3VYaEMwVDA/view?usp=sharing](https://drive.google.com/file/d/0B6VgjAr4t_oTQXN2Y3VYaEMwVDA/view?usp=sharing). Download and unzip them to the root folder of the code. You can run them by executing, for example, `start('cache_dir','./cache_iccv_cub200','cnn_dir','./cnn_finetuning/vgg19/','crop_size',224,'basedir','/home/simon/Datasets/CUB_200_2011/')`.

## License 
The Part Constellation Models Framework by [Marcel Simon](http://www.inf-cv.uni-jena.de/simon.html) and [Erik Rodner](http://www.inf-cv.uni-jena.de/rodner.html) is licensed under the non-commercial license [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/). For usage beyond the scope of this license, please contact [Marcel Simon](http://www.inf-cv.uni-jena.de/simon.html).
