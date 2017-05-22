#include "caffe/net.hpp"
#include "caffe/layers/ogn_prop_layer.hpp"
#include "caffe/layers/ogn_conv_layer.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

using namespace std;

template <typename Dtype>
void OGNPropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	_done_initial_reshape = false;
	_done_building_graph = false;
}

template <typename Dtype>
void OGNPropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const int num = bottom[0]->shape(0);
    const int channels = bottom[0]->shape(1);

	if(!_done_initial_reshape)
	{
		_num_output_pixels = 1;
		_done_initial_reshape = true;
	}
	else
	{
		if(!_done_building_graph)
		{
			bool in_current_block = false;
			NetworkGraph graph;
	
			for(typename Net<Dtype>::const_iterator it=this->parent_net()->begin(); it!=this->parent_net()->end(); it++)
			{
				if(it->get() == this)
				{
					in_current_block = true;
					continue;
				}
	
				string l_type = (*it)->type();
				if(in_current_block)
				{
					if(l_type == "OGNConv")
					{
						boost::shared_ptr<OGNConvLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNConvLayer<Dtype> >((*it));
			
						int filter_size = l_ptr->layer_param().ogn_conv_param().filter_size();
						bool is_deconv = l_ptr->layer_param().ogn_conv_param().is_deconv();
						graph.add_layer(is_deconv, filter_size);
					}
				}
			}
	
			_nbh_prop_size = graph.compute_neighborhood_size();
			_done_building_graph = true;
		}
		compute_pixel_propagation(bottom, top);
	}

	if(!_num_output_pixels) _num_output_pixels = 1;

	vector<int> shape_features;
    shape_features.push_back(num); shape_features.push_back(channels); shape_features.push_back(_num_output_pixels);
    top[0]->Reshape(shape_features);
}

template <typename Dtype>
void OGNPropLayer<Dtype>::compute_pixel_propagation(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	this->_octree_keys.clear();
	this->_octree_prop.clear();

	_num_output_pixels = 0;
    const Dtype* input_values = bottom[1]->cpu_data();
    const int num = bottom[0]->shape(0);
    const int pixels = bottom[0]->shape(2);

    for(int bt=0; bt<num; bt++)
    {
    	int counter_top = 0;
    	GeneralOctree<int> octree_keys;
    	GeneralOctree<int> octree_prop;

    	std::string key_layer_name = this->layer_param_.ogn_prop_param().key_layer();
    	boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    	boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

    	for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(bt).begin(); it!=l_ptr->get_keys_octree(bt).end(); it++)
    	{
    		if(l_ptr->get_prop_octree(bt).get_value(it->first) != PROP_TRUE) continue;

    		SignalType v;
    		OGNPropParameter_PropagationMode prop_mode = this->layer_param().ogn_prop_param().prop_mode();

			if(prop_mode == OGNPropParameter_PropagationMode_PROP_PRED)
			{
				Dtype max_val = 0;
				for(int cl=0; cl<OGN_NUM_CLASSES; cl++)
				{
					Dtype val = input_values[bt * OGN_NUM_CLASSES * pixels + cl * pixels + it->second];
					if(val > max_val)
					{
						max_val = val;
						v = cl;
					}
				}
			}
			else if(prop_mode == OGNPropParameter_PropagationMode_PROP_KNOWN)
			{
				v = input_values[bt * pixels + it->second];
			}

			if(v == CLASS_MIXED)
			{
				if(_nbh_prop_size > 1)
				{
					std::vector<KeyType> neighbors = l_ptr->get_keys_octree(bt).get_neighbor_keys(it->first, _nbh_prop_size);
					for(int i=0; i<neighbors.size(); i++)
                    {
                    	if(neighbors[i] != GeneralOctree<int>::INVALID_KEY())
                        {
                            KeyType nbh_key = neighbors[i];
                            if(octree_keys.get_value(nbh_key) == -1)
							{
								octree_keys.add_element(nbh_key, counter_top);
								octree_prop.add_element(nbh_key, PROP_FALSE);
								counter_top++;
							}
                        }
                    }
				}

				if(octree_keys.get_value(it->first) == -1)
				{
					octree_keys.add_element(it->first, counter_top);
					octree_prop.add_element(it->first, PROP_TRUE);
					counter_top++;
				}
				else
				{
					octree_prop.add_element(it->first, PROP_TRUE);
				}
			}
    	}

    	if(counter_top > _num_output_pixels) _num_output_pixels = counter_top;
    	this->_octree_keys.push_back(octree_keys);
    	this->_octree_prop.push_back(octree_prop);
    }
}

template <typename Dtype>
void OGNPropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const Dtype* input_features = bottom[0]->cpu_data();
	Dtype* output_features = top[0]->mutable_cpu_data();

	const int num = bottom[0]->shape(0);
    const int channels = bottom[0]->shape(1);
    const int input_pixels = bottom[0]->shape(2);

    memset(output_features, 0, sizeof(Dtype)*num*channels*_num_output_pixels);

    std::string key_layer_name = this->layer_param_.ogn_prop_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

    for(int bt=0; bt<num; bt++)
    {
        for(typename GeneralOctree<int>::iterator it=this->_octree_keys[bt].begin(); it!=this->_octree_keys[bt].end(); it++)
        {
            for(int ch=0; ch<channels; ch++)
            {
                output_features[bt * channels * _num_output_pixels + ch * _num_output_pixels + it->second] =
                    input_features[bt * channels * input_pixels + ch * input_pixels + l_ptr->get_keys_octree(bt).get_value(it->first)];
            }
        }
    }

}

template <typename Dtype>
void OGNPropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const int num = bottom[0]->shape(0);
    const int channels = bottom[0]->shape(1);
    const int pixels = bottom[0]->shape(2);

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    memset(bottom_diff, 0, sizeof(Dtype)*num*channels*pixels);

    std::string key_layer_name = this->layer_param_.ogn_prop_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OGNLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OGNLayer<Dtype> >(base_ptr);

    for(int bt=0; bt<num; bt++)
    {
        for(typename GeneralOctree<int>::iterator it=this->_octree_keys[bt].begin(); it!=this->_octree_keys[bt].end(); it++)
        {
            for(int ch=0; ch<channels; ch++)
            {
                bottom_diff[bt * channels * pixels + ch * pixels + l_ptr->get_keys_octree(bt).get_value(it->first)] +=
                    top_diff[bt * channels * _num_output_pixels + ch * _num_output_pixels + it->second];
            }
        }
    }

}

INSTANTIATE_CLASS(OGNPropLayer);
REGISTER_LAYER_CLASS(OGNProp);

}  // namespace caffe
