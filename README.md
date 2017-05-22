#OGN

Source code accompanying the paper "Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs" by M. Tatarchenko, A. Dosovitskiy and T. Brox https://lmb.informatik.uni-freiburg.de/people/tatarchm/ogn/.

## Build
For compilation instructions please refer to the [http://caffe.berkeleyvision.org/installation.html](official) or [https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide](unofficial) Caffe cmake build guidelines.

## Usage
You can download the datasets and example models from [](here). Run one of the scripts (train_known.sh, train_pred.sh or test.sh) from the corresponding experiment folder. You should have the caffe executable in your $PATH.

## Visualization
We provide a python script for visualizing the .ot files in Blender. To use it, run
	
	$ blender -P $CAFFE_ROOT/python/rendering/render_model.py your_model.ot

## License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. When using the code in your research work, please cite the following paper:

 @article{ogn2017,
  author    = {Maxim Tatarchenko and Alexey Dosovitskiy and Thomas Brox},
  title     = {Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs},
  journal   = {CoRR},
  volume    = {abs/1703.09438},
  year      = {2017}
  }