#ifndef OGN_LOSS_PREP_LAYER_HPP_
#define OGN_LOSS_PREP_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OGNLossPrepLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNLossPrepLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNLossPrep"; }
  // virtual inline int ExactNumBottomBlobs() const { return 1; }
  // virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  //OGN_LOSS_PREP_LAYER_HPP_
