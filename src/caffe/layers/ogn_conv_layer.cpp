#include "caffe/layers/ogn_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();
	const int filter_size = this->layer_param_.ogn_conv_param().filter_size();
	_num_output_channels = this->layer_param_.ogn_conv_param().output_channels();
	_num_input_channels = bottom[0]->shape(1);

	this->blobs_.resize(2);

	if (is_deconv)
    {
        _weight_shape.push_back(_num_input_channels);
        _weight_shape.push_back(_num_output_channels);
        _weight_shape.push_back(filter_size);
        _weight_shape.push_back(filter_size);
        _weight_shape.push_back(filter_size);
    }
    else
    {
        _weight_shape.push_back(_num_output_channels);
        _weight_shape.push_back(_num_input_channels);
        _weight_shape.push_back(filter_size);
        _weight_shape.push_back(filter_size);
        _weight_shape.push_back(filter_size);
    }

    _bias_shape.push_back(_num_output_channels);

    this->blobs_[0].reset(new Blob<Dtype>(_weight_shape));
    this->blobs_[1].reset(new Blob<Dtype>(_bias_shape));

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.ogn_conv_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.ogn_conv_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
}

template <typename Dtype>
void OGNConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	_batch_size = bottom[0]->shape(0);
    _num_input_pixels = bottom[0]->shape(2);

    const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();
    const int filter_size = this->layer_param_.ogn_conv_param().filter_size();

    if(is_deconv) _num_output_pixels = 8 * _num_input_pixels;
    else _num_output_pixels = _num_input_pixels;

    vector<int> features_shape;
    features_shape.push_back(_batch_size);
    features_shape.push_back(_num_output_channels);
    features_shape.push_back(_num_output_pixels);
    top[0]->Reshape(features_shape);

    _col_buffer_shape.clear();
    _col_buffer_shape.push_back(_weight_shape[1] * filter_size * filter_size * filter_size);
    if(is_deconv) _col_buffer_shape.push_back(_num_input_pixels);
    else _col_buffer_shape.push_back(_num_output_pixels);

    _col_buffer.Reshape(_col_buffer_shape);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::propagate_keys_cpu()
{
	const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();

	this->_octree_keys.clear();
    this->_octree_prop.clear();

    std::string key_layer_name = this->layer_param_.ogn_conv_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

    for (int n = 0; n < _batch_size; ++n)
    {
        GeneralOctree<int> octree_keys;
        GeneralOctree<int> octree_prop;

        if(is_deconv)
        {
            int output_counter = 0;
            for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(n).begin(); it!=l_ptr->get_keys_octree(n).end(); it++)
            {
                KeyType key = it->first;
                for(int i=0; i<8; i++)
                {
                    unsigned int new_key = (key << 3) | i;
                    octree_keys.add_element(new_key, output_counter);
                    octree_prop.add_element(new_key, l_ptr->get_prop_octree(n).get_value(key));
                    output_counter++;
                }
            }
            this->_octree_keys.push_back(octree_keys);
            this->_octree_prop.push_back(octree_prop);
        }
        else
        {
            this->_octree_keys.push_back(l_ptr->get_keys_octree(n));
            this->_octree_prop.push_back(l_ptr->get_prop_octree(n));
        }
    }
}

template <typename Dtype>
void OGNConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	propagate_keys_cpu();

	const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();
	for (int n=0; n<_batch_size; n++)
    {
    	int num_elements = this->_octree_keys[n].num_elements();
        if(is_deconv)
        {
        	resize_computation_buffers_cpu(num_elements);
            backward_cpu_gemm(bottom[0]->cpu_data() + n * _num_input_channels * _num_input_pixels, this->blobs_[0]->cpu_data(),
                    _col_buffer.mutable_cpu_data());
            col2im_octree_cpu(n, bottom, top);
            forward_cpu_bias(top[0]->mutable_cpu_data() + n * _num_output_channels * _num_output_pixels, this->blobs_[1]->cpu_data());
        }
        else
        {
            resize_computation_buffers_cpu(num_elements);
            im2col_octree_cpu(n, bottom, top);
            forward_cpu_gemm(this->blobs_[0]->cpu_data(), _col_buffer.mutable_cpu_data(),
                top[0]->mutable_cpu_data() + n * _num_output_channels * _num_output_pixels);
            forward_cpu_bias(top[0]->mutable_cpu_data() +
                n * _num_output_channels * _num_output_pixels, this->blobs_[1]->cpu_data());
        }
    }

}

template <typename Dtype>
void OGNConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();

    for (int n = 0; n < _batch_size; ++n)
    {
        if(is_deconv)
        {
            resize_computation_buffers_cpu(this->_octree_keys[n].num_elements());
            backward_cpu_bias(bias_diff, top_diff + n * _num_output_pixels * _num_output_channels);
            im2col_octree_cpu(n, bottom, top);
            weight_cpu_gemm(_col_buffer.cpu_data(), bottom[0]->cpu_data() + n * _num_input_channels * _num_input_pixels, weight_diff);
            forward_cpu_gemm(this->blobs_[0]->cpu_data(), _col_buffer.cpu_data(), bottom[0]->mutable_cpu_diff() + n * _num_input_pixels * _num_input_channels);
        }
        else
        {
            resize_computation_buffers_cpu(this->_octree_keys[n].num_elements());
            im2col_octree_cpu(n, bottom, top);
            weight_cpu_gemm(_col_buffer.mutable_cpu_data(),
                top_diff + n * _num_output_channels * _num_output_pixels, weight_diff);
            backward_cpu_bias(bias_diff, top_diff + n * _num_output_channels * _num_output_pixels);
            backward_cpu_gemm(top_diff + n * _num_output_channels * _num_output_pixels,
                this->blobs_[0]->cpu_data(), _col_buffer.mutable_cpu_data());
            col2im_octree_cpu(n, bottom, top);
        }
    }

}



template <typename Dtype>
void OGNConvLayer<Dtype>::resize_computation_buffers_cpu(int batch_num_pixels)
{
    vector<int> bias_multiplier_shape;
    bias_multiplier_shape.push_back(_num_output_pixels); bias_multiplier_shape.push_back(1);
    _bias_multiplier.Reshape(bias_multiplier_shape);
    caffe_set(_num_output_pixels, Dtype(0), _bias_multiplier.mutable_cpu_data());
    caffe_set(batch_num_pixels, Dtype(1), _bias_multiplier.mutable_cpu_data());
    memset(_col_buffer.mutable_cpu_data(), 0, sizeof(Dtype)*_col_buffer_shape[0]*_col_buffer_shape[1]);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::backward_cpu_gemm(const Dtype* top_diff, const Dtype* weights, Dtype* col_buff)
{
	const int filter_size = this->layer_param_.ogn_conv_param().filter_size();

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * filter_size * filter_size * filter_size,
        _col_buffer_shape[1], _weight_shape[0],
        (Dtype)1., weights, top_diff,
        (Dtype)0., col_buff);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::forward_cpu_gemm(const Dtype* weights, const Dtype* input, Dtype* output)
{
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
        _col_buffer_shape[1], _col_buffer_shape[0],
        (Dtype)1., weights, input,
        (Dtype)0., output);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias)
{
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _bias_shape[0], _num_output_pixels, 1,
                          (Dtype)1., bias, _bias_multiplier.cpu_data(),
                          (Dtype)1., output);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input)
{
    caffe_cpu_gemv<Dtype>(CblasNoTrans, _num_output_channels, _num_output_pixels, 1.,
      input, _bias_multiplier.cpu_data(), (Dtype)1., bias);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights)
{
    const Dtype* col_buff = input;
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                          _col_buffer_shape[0], _col_buffer_shape[1],
                          (Dtype)1., output, col_buff, (Dtype)1., weights);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::col2im_octree_cpu(int batch_ind, const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	const int filter_size = this->layer_param_.ogn_conv_param().filter_size();
	const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();

    std::string key_layer_name = this->layer_param_.ogn_conv_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

    Dtype* output_arr;
    int output_rows, output_cols;
    if(is_deconv)
    {
        output_arr = top[0]->mutable_cpu_data();
        output_rows = _num_output_channels;
        output_cols = _num_output_pixels;
    }
    else
    {
        output_arr = bottom[0]->mutable_cpu_diff();
        output_rows = _num_input_channels;
        output_cols = _num_input_pixels;
    }

    if(!batch_ind) memset(output_arr, 0, sizeof(Dtype) * _batch_size * output_rows * output_cols);

    for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(batch_ind).begin(); it!=l_ptr->get_keys_octree(batch_ind).end(); it++)
    {
        KeyType key = it->first;

        if(is_deconv) key = key << 3;

        std::vector<KeyType> neighbors = this->_octree_keys[batch_ind].get_neighbor_keys(key, filter_size);
        for(int ch=0; ch<output_rows; ch++)
        {
            for(int el=0; el<filter_size*filter_size*filter_size; el++)
            {
                int col_buff_ind = ch * neighbors.size() * _col_buffer_shape[1] + el * _col_buffer_shape[1] + it->second;

                if(neighbors[el] != GeneralOctree<int>::INVALID_KEY())
                {
                    unsigned int nbh_key = neighbors[el];
                    int feature_ind = batch_ind * output_rows * output_cols +
                                  ch * output_cols + this->_octree_keys[batch_ind].get_value(nbh_key);

                    output_arr[feature_ind] += _col_buffer.mutable_cpu_data()[col_buff_ind];
                }
            }
        }
    }

    memset(_col_buffer.mutable_cpu_data(), 0, sizeof(Dtype)*_col_buffer_shape[0]*_col_buffer_shape[1]);
}

template <typename Dtype>
void OGNConvLayer<Dtype>::im2col_octree_cpu(int batch_ind, const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	const int filter_size = this->layer_param_.ogn_conv_param().filter_size();
	const bool is_deconv = this->layer_param_.ogn_conv_param().is_deconv();

    Dtype* col_buff = _col_buffer.mutable_cpu_data();

    int input_rows, input_cols;
    if(is_deconv)
    {
        input_rows = _num_output_channels;
        input_cols = _num_output_pixels;
    }
    else
    {
        input_rows = _num_input_channels;
        input_cols = _num_input_pixels;
    }

    std::string key_layer_name = this->layer_param_.ogn_conv_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

    for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(batch_ind).begin(); it!=l_ptr->get_keys_octree(batch_ind).end(); it++)
    {
        KeyType key = it->first;
        if(is_deconv) key = key << 3;

        if(key)
        {
            std::vector<KeyType> neighbors = this->_octree_keys[batch_ind].get_neighbor_keys(key, filter_size);

            for(int ch=0; ch<input_rows; ch++)
            {
                for(int el=0; el<neighbors.size(); el++)
                {
                    int col_buff_ind = ch * neighbors.size() * _col_buffer_shape[1] + el * _col_buffer_shape[1] + it->second;

                    if(neighbors[el] != GeneralOctree<int>::INVALID_KEY())
                    {
                        KeyType nbh_key = neighbors[el];
                        int feature_ind = batch_ind * input_rows * input_cols +
                                      ch * input_cols + this->_octree_keys[batch_ind].get_value(nbh_key);
                        if(is_deconv) col_buff[col_buff_ind] = top[0]->cpu_diff()[feature_ind];
                        else col_buff[col_buff_ind] = bottom[0]->cpu_data()[feature_ind];
                    }
                    else
                    {
                        col_buff[col_buff_ind] = 0;
                    }
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(OGNConvLayer);
#endif

INSTANTIATE_CLASS(OGNConvLayer);
REGISTER_LAYER_CLASS(OGNConv);

}  // namespace caffe
