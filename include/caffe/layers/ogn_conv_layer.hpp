#ifndef OGN_CONV_LAYER_HPP_
#define OGN_CONV_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"

namespace caffe {

template <typename Dtype>
class OGNConvLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNConvLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNConv"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void propagate_keys_cpu();
  void resize_computation_buffers_cpu(int batch_num_pixels);
  void backward_cpu_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* col_buff);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void col2im_octree_cpu(int batch_ind, const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void im2col_octree_cpu(int batch_ind, const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void forward_cpu_gemm(const Dtype* weights, const Dtype* input, Dtype* output);

  vector<int> _weight_shape;
  vector<int> _bias_shape;
  vector<int> _col_buffer_shape;

  Blob<Dtype> _col_buffer;
  Blob<Dtype> _bias_multiplier;

  int _num_input_pixels;
  int _num_output_pixels;
  int _num_output_channels;
  int _num_input_channels;
  int _batch_size;

};

}  // namespace caffe

#endif  // OGN_CONV_LAYER_HPP_
