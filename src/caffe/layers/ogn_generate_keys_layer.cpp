#include "caffe/layers/ogn_generate_keys_layer.hpp"

#include "image_tree_tools/image_tree_tools.h"

namespace caffe {

template <typename Dtype>
void OGNGenerateKeysLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    this->_level = 0;
}

template <typename Dtype>
void OGNGenerateKeysLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    for(int i=2; i<5; i++)
    {
        int dim = bottom[0]->shape(i);
        int level = ceil(log(dim) / log(2));
        if(level > this->_level) this->_level = level;
    }

    this->_octree_keys.clear();
    this->_octree_prop.clear();

    int batch_size = bottom[0]->shape(0);
    int xsize = bottom[0]->shape(2);
    int ysize = bottom[0]->shape(3);
    int zsize = bottom[0]->shape(4);

    for(int bt=0; bt<batch_size; bt++)
    {
        GeneralOctree<int> octree_keys;
        GeneralOctree<int> octree_prop;
        for(int x=0; x<xsize; x++)
        {
            for(int y=0; y<ysize; y++)
            {
                for(int z=0; z<zsize; z++)
                {
                    OctreeCoord c;
                    c.x = x; c.y = y; c.z = z; c.l = this->_level;
                    KeyType key = Octree::compute_key(c);

                    octree_keys.add_element(key, x*ysize*zsize + y*zsize + z);
                    octree_prop.add_element(key, 1);
                }
            }
        }
        this->_octree_keys.push_back(octree_keys);
        this->_octree_prop.push_back(octree_prop);
    }
}

template <typename Dtype>
void OGNGenerateKeysLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OGNGenerateKeysLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    LOG(FATAL) << "Backward not implemented";
}

INSTANTIATE_CLASS(OGNGenerateKeysLayer);
REGISTER_LAYER_CLASS(OGNGenerateKeys);

}  // namespace caffe
