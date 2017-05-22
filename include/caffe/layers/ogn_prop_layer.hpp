#ifndef OGN_PROP_LAYER_HPP_
#define OGN_PROP_LAYER_HPP_

#include "caffe/layers/ogn_layer.hpp"

#include <set>

namespace caffe {

using namespace std;

/// A layer for propagating the features of the cells that are "mixed".
/// Information about the state of a cell can either be taken from the ground truth tree, or from the prediction.
/// Depending on the filter sizes in subsequent OGNConv layers, also propagates features of the
/// neighboring cells needed for computations.
template <typename Dtype>
class OGNPropLayer : public OGNLayer<Dtype> {
 public:
  explicit OGNPropLayer(const LayerParameter& param)
      : OGNLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OGNProp"; }

 protected:

  void compute_pixel_propagation(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  bool _done_initial_reshape;
  bool _done_building_graph;

  int _nbh_prop_size;
  int _num_output_pixels;

};


/// A class for representing a NN graph. Used to
/// calculate the number of neighbors that need to be propagated
/// in order to compute convolutions in the later layers.
class NetworkGraph
{

private:

  struct LayerEntry
  {
    vector<int> elements;
    map<int, vector<int> > bottom_links;
    map<int, vector<int> > top_links;
  };

  vector<LayerEntry> _layers;

public:

  NetworkGraph()
  {
    LayerEntry ent;
    for(int i=0; i<100; i++) ent.elements.push_back(i);
    _layers.push_back(ent);
  }

  void add_layer(bool is_deconv, int filter_size)
  {
    LayerEntry ent;
    int len = _layers[_layers.size()-1].elements.size();
    int min_val = -(filter_size/2);
    if(filter_size % 2 == 0) min_val++;
    int max_val = filter_size/2;
    if(is_deconv)
    {
      for(int i=0; i<len; i++)
      {
        ent.elements.push_back(i*2);
        ent.elements.push_back(i*2+1);
      }
      for(int i=0; i<len; i++)
      {
        vector<int> top_links;
        for(int j=min_val; j<=max_val; j++)
        {
          int link = i*2 + j;
          if(link>=0 && link<len*2)
          {
            top_links.push_back(link);
            if(ent.bottom_links.find(link) == ent.bottom_links.end())
              ent.bottom_links[link] = vector<int>();
            ent.bottom_links[link].push_back(i);
          }
        }
        _layers[_layers.size()-1].top_links[i] = top_links;
      }
    }
    else
    {
      for(int i=0; i<len; i++)
      {
        ent.elements.push_back(i);
      }
      for(int i=0; i<len; i++)
      {
        vector<int> bottom_links;
        for(int j=min_val; j<=max_val; j++)
        {
          int link = i + j;
          if(link>=0 && link<len)
          {
            bottom_links.push_back(link);

            if(_layers[_layers.size()-1].top_links.find(link) == _layers[_layers.size()-1].top_links.end())
              _layers[_layers.size()-1].top_links[link] = vector<int>();
            _layers[_layers.size()-1].top_links[link].push_back(i);
          }
        }
        ent.bottom_links[i] = bottom_links;
      }
    }
    _layers.push_back(ent);
  }

  int compute_neighborhood_size()
  {
    set<int> curr_elements;
    
    int num_outputs = _layers[_layers.size()-1].elements.size() / _layers[0].elements.size();
    int mid = _layers[_layers.size()-1].elements[_layers[_layers.size()-1].elements.size()/2];
    for(int i=mid; i<mid+num_outputs; i++) curr_elements.insert(i);

    for(int l=_layers.size()-1; l>0; l--)
    {
      set<int> next_elements;
      for(set<int>::iterator it=curr_elements.begin(); it!=curr_elements.end(); it++)
      {
        for(int i=0; i<_layers[l].bottom_links[*it].size(); i++)
        {
          next_elements.insert(_layers[l].bottom_links[*it][i]);
        }
      }
      curr_elements = next_elements;
    }

    return curr_elements.size();
  }

};

}  // namespace caffe

#endif  // OGN_PROP_LAYER_HPP_
