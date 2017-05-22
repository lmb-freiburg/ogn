#ifndef OGN_LAYER_HPP_
#define OGN_LAYER_HPP_

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
class OGNLayer : public Layer<Dtype> {

public:
  explicit OGNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  //TODO: make these references constant
  GeneralOctree<int>& get_keys_octree(int batch_ind)
  {
      return _octree_keys[batch_ind];
  }

  GeneralOctree<int>& get_prop_octree(int batch_ind)
  {
      return _octree_prop[batch_ind];
  }

  int get_level() {return _level;}

protected:

  std::vector<GeneralOctree<int> > _octree_keys;
  std::vector<GeneralOctree<int> > _octree_prop;
  int _level;

};

}  // namespace caffe

#endif  // OGN_LAYER_HPP_
