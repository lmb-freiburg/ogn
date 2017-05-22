#include "caffe/layers/ogn_output_layer.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

using namespace std;

template <typename Dtype>
void OGNOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    _output_num = 0;
    _done_initial_reshape = false;
}

template <typename Dtype>
void OGNOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    if(!_done_initial_reshape)
    {
        _done_initial_reshape = true;
        return;
    }

    int batch_size = bottom[0]->shape(0);
    std::string output_path = this->layer_param_.ogn_output_param().output_path();
    int key_layer_size = this->layer_param_.ogn_output_param().key_layer_size();

    if(key_layer_size != bottom.size())
            LOG(FATAL) << "Number of key layers does not match the number of input blobs.";

    for(int bt=0; bt<batch_size; bt++)
    {
        Octree octr;

        for(int i=0; i<key_layer_size; i++)
        {   
            std::string key_layer_name = this->layer_param_.ogn_output_param().key_layer(i);
            boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
            boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);
    
            for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(bt).begin(); it!=l_ptr->get_keys_octree(bt).end(); it++)
            {
                SignalType value;
                //ground truth case
                if(bottom[0]->num_axes() == 2)
                {
                    int num_pixels = bottom[i]->shape(1);
                    int value_index = bt * num_pixels + it->second;
                    value = (SignalType)bottom[i]->cpu_data()[value_index];
                }
                //prediction case
                else
                {
                    int num_pixels = bottom[i]->shape(2);
                    Dtype max_val = 0;
                    for(int cl=0; cl<OGN_NUM_CLASSES; cl++)
                    {
                        Dtype val = bottom[i]->cpu_data()[bt * OGN_NUM_CLASSES * num_pixels + cl * num_pixels + it->second];
                        if(val > max_val)
                        {
                            max_val = val;
                            value = cl;
                        }
                    }
                }
                if(value != CLASS_MIXED) octr.add_element(it->first, value);
            }
        }

        if(output_path.length() > 0)
        {
            std::stringstream ss;
            ss << output_path << std::setfill('0') << std::setw(4) << _output_num++ << ".ot";
            std::string output_file_name = ss.str();
            octr.to_file(output_file_name);
        }
    }
}

template <typename Dtype>
void OGNOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OGNOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(OGNOutputLayer);
REGISTER_LAYER_CLASS(OGNOutput);

}  // namespace caffe
