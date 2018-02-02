# OGN

<p align="center"> 
<img src="https://github.com/mtatarchenko/ogn/blob/master/thumbnail.png">
</p>

Source code accompanying the paper ["Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs"](https://lmb.informatik.uni-freiburg.de/people/tatarchm/ogn/) by M. Tatarchenko, A. Dosovitskiy and T. Brox. The implementation is based on [Caffe](http://caffe.berkeleyvision.org/), and extends the basic framework by providing layers for octree-specific features.

## Build
For compilation instructions please refer to the [official](http://caffe.berkeleyvision.org/installation.html) or [unofficial](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide) CMake build guidelines for Caffe. Makefile build is not supported.

## Data
Octrees are stored as text-based serialized std::map containers. The provided utility (tools/ogn_converter) can be used to convert [binvox](http://minecraft.gamepedia.com/Programs_and_editors/Binvox) voxel grids into octrees. Three of the datasets used in the paper (ShapeNet-cars, FAUST and BlendSwap) can be downloaded from [here](http://lmb.informatik.uni-freiburg.de/data/ogn/data.zip). For ShapeNet-all, we used the voxelizations(ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz) and the renderings(ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz) provided by Choy et al. for their [3D-R<sup>2</sup>N<sup>2</sup>](https://github.com/chrischoy/3D-R2N2) framework.

## Usage
Example models can be downloaded from [here](http://lmb.informatik.uni-freiburg.de/data/ogn/examples.zip). Run one of the scripts (train_known.sh, train_pred.sh or test.sh) from the corresponding experiment folder. You should have the caffe executable in your $PATH.

## Visualization
There is a python script for visualizing .ot files in Blender. To use it, run
	
	$ blender -P $CAFFE_ROOT/python/rendering/render_model.py your_model.ot

## License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. When using the code in your research work, please cite the following paper:
```
 @InProceedings{ogn2017,
  author       = "M. Tatarchenko and A. Dosovitskiy and T. Brox",
  title        = "Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2017/TDB17b"
}
 
```
