#ifndef OGN_OUTPUT_LAYER_HPP_
#define OGN_OUTPUT_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/// Assembles an octree based on the network predictions,
/// and outputs it to file.
template <typename Dtype>
class OGNOutputLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNOutputLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNOutput"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int _output_num;
  bool _done_initial_reshape;

};

}  // namespace caffe

#endif  //OGN_OUTPUT_LAYER_HPP_
