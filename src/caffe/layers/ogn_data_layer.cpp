#include "caffe/layers/ogn_data_layer.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void OGNDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    _model_counter = 0;
    _done_initial_reshape = false;
    load_data_from_disk();
}

template <typename Dtype>
void OGNDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const int batch_size = this->layer_param_.ogn_data_param().batch_size();

    vector<int> values_shape;
    vector<int> labels_shape;
    labels_shape.push_back(batch_size);

    if(!_done_initial_reshape)
    {
        values_shape.push_back(batch_size); values_shape.push_back(1);
        _done_initial_reshape = true;
    }
    else
    {
        vector<int> batch_elements;
        for(int bt=0; bt<batch_size; bt++)
        {
            if(bottom.size() == 0)
            {
                batch_elements.push_back(_model_counter++);
                if(_model_counter == _file_names.size()) _model_counter = 0;
            }
            else
            {
                batch_elements.push_back(bottom[0]->cpu_data()[bt]);
            }
        }
        int num_elements = select_next_batch_models(batch_elements);
        values_shape.push_back(batch_size); values_shape.push_back(num_elements);
    }

    top[0]->Reshape(values_shape);
    top[1]->Reshape(labels_shape);
}

template <typename Dtype>
void OGNDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    this->_octree_keys.clear();
    const int batch_size = this->layer_param_.ogn_data_param().batch_size();
    int num_elements = top[0]->shape(1);

    Dtype* top_values = top[0]->mutable_cpu_data();
    Dtype* top_labels = top[1]->mutable_cpu_data();

    memset(top_values, 0, sizeof(Dtype) * top[0]->count());
    memset(top_labels, 0, sizeof(Dtype) * top[1]->count());

    for(int bt=0; bt<batch_size; bt++)
    {
        GeneralOctree<int> octree_keys;
        int counter = 0;
        for(Octree::iterator it=_batch_octrees[bt].begin(); it!=_batch_octrees[bt].end(); it++)
        {
            int top_index = bt * num_elements + counter;
            top_values[top_index] = (Dtype)(it->second);
            octree_keys.add_element(it->first, counter);
            counter++;
        }
        top_labels[bt] = _batch_labels[bt];
        this->_octree_keys.push_back(octree_keys);
    }
}

template <typename Dtype>
void OGNDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}

template <typename Dtype>
int OGNDataLayer<Dtype>::select_next_batch_models(vector<int> labels)
{
    const bool preload_data = this->layer_param_.ogn_data_param().preload_data();
    int num_elements = 0;
    _batch_octrees.clear();
    _batch_labels.clear();

    for(int bt=0; bt<labels.size(); bt++)
    {
        int len = 0;
        _batch_labels.push_back(labels[bt]);
        if(preload_data)
        {
            _batch_octrees.push_back(_octrees[labels[bt]]);
            len = _octrees[labels[bt]].num_elements();
        }
        else
        {
            Octree tree;
            tree.from_file(_file_names[labels[bt]]);
            _batch_octrees.push_back(tree);
            len = tree.num_elements();
        }
        if(len > num_elements) num_elements = len;
    }
    return num_elements;
}

template <typename Dtype>
void OGNDataLayer<Dtype>::load_data_from_disk()
{
    cout << "Loading training data from disk..." << endl;
    const string source = this->layer_param_.ogn_data_param().source();
    const bool preload_data = this->layer_param_.ogn_data_param().preload_data();

    ifstream infile(source.c_str());
    string name;
    int counter = 0;
    while(infile >> name)
    {
        _file_names.push_back(name);
        if(preload_data)
        {
            Octree tree;
            tree.from_file(name);
            _octrees.push_back(tree);
            cout << name << endl;
        }
        counter++;
    }

    std::cout << "Done." << std::endl;
}

INSTANTIATE_CLASS(OGNDataLayer);
REGISTER_LAYER_CLASS(OGNData);

}  // namespace caffe
