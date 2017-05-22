#ifndef OCTREE_H_
#define OCTREE_H_

#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <tr1/unordered_map>

#include <math.h>

#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "zindex.h"
#include "voxel_grid.h"
#include "common_util.h"

struct OctreeCoord
{
    int x;
    int y;
    int z;
    int l;
};

template <class VALUE>
class GeneralOctree
{

public:
  typedef unsigned int KEY;

private:
  typedef std::tr1::unordered_map<KEY, VALUE> HashTable;

  HashTable _hash_table;
  int _max_level;

public:

  static int MIN_LEVEL() { return 0; }
  static int MAX_LEVEL() { return (sizeof(KEY) * 8) / 3; }
  static KEY INVALID_KEY() { return 0; }
  static bool IS_VALID_COORD(const OctreeCoord& c)
  {
    if( c.l > MAX_LEVEL() ||
        c.x < 0 || c.x >= (int(1) << c.l) ||
        c.y < 0 || c.y >= (int(1) << c.l) ||
        c.z < 0 || c.z >= (int(1) << c.l) ) return false;
    return true;
  }
  static bool IS_VALID_KEY(const KEY& key)
  {
    if(key == INVALID_KEY()) return false;
    int lz = __builtin_clz(key) - 1;
    if(lz % 3) return false;
    return true;
  }

  GeneralOctree(int max_level = -1)
  {
      _max_level = max_level;
  }

  typedef typename HashTable::iterator iterator;
  typedef typename HashTable::const_iterator const_iterator;
  iterator begin() { return _hash_table.begin(); }
  iterator end() { return _hash_table.end(); }

  static int resolution_from_level(int level)
  {
      return pow(2, level);
  }

  static int compute_level(const KEY& key)
  {
      return (MAX_LEVEL() * 3 - __builtin_clz(key) + 1) / 3;
  }

  static KEY compute_key(const OctreeCoord& c)
  {
      if(!IS_VALID_COORD(c)) return INVALID_KEY();
      return morton_3d(KEY(c.x), KEY(c.y), KEY(c.z)) | (KEY(1) << 3 * c.l);
  }

  static OctreeCoord compute_coord(KEY key)
  {
    OctreeCoord c;
    c.l = compute_level(key);

    KEY x,y,z;
    inverse_morton_3d(x, y, z, key & ~(KEY(1) << c.l * 3));
    c.x = x;
    c.y = y;
    c.z = z;
    return c;
  }

  int num_elements() {return _hash_table.size();}

  void add_element(KEY key, VALUE value)
  {
      _hash_table[key] = value;
  }

  std::pair<KEY, VALUE> get_element(int i)
  {
      typename std::map<KEY, VALUE>::iterator it = _hash_table.begin();
      std::advance(it, i);
      return *it;
  }

  std::vector<KEY> get_neighbor_keys(KEY key, int nbh_size)
  {
    std::vector<KEY> ret;

    OctreeCoord c = compute_coord(key);
    int res = resolution_from_level(c.l);

    for(int i=0; i<(_max_level - c.l); i++) res /= 2;

    int min_ind = -nbh_size / 2;
    int max_ind = nbh_size / 2;
    if(nbh_size % 2 == 0) min_ind += 1;

    for(int i=min_ind; i<=max_ind; i++)
    {
      for(int j=min_ind; j<=max_ind; j++)
      {
        for(int k=min_ind; k<=max_ind; k++)
        {
          if(c.x+i < 0 || c.x+i >= res)      ret.push_back(INVALID_KEY());
          else if(c.y+j < 0 || c.y+j >= res) ret.push_back(INVALID_KEY());
          else if(c.z+k < 0 || c.z+k >= res) ret.push_back(INVALID_KEY());
          else
          {
            OctreeCoord nb_c;
            nb_c.x = c.x+i; nb_c.y = c.y+j; nb_c.z = c.z+k; nb_c.l = c.l;
            KEY new_code = compute_key(nb_c);
            //typename std::map<KEY, VALUE>::iterator it = _hash_table.find(new_code);
            typename std::tr1::unordered_map<KEY, VALUE>::iterator it = _hash_table.find(new_code);

            if(it != _hash_table.end()) ret.push_back(it->first);
            else ret.push_back(INVALID_KEY());
          }
        }
      }
    }

    return ret;
  }

  VALUE get_value(KEY key, bool use_vg_info = false)
  {
      typename HashTable::iterator it = _hash_table.find(key);
      if(it != _hash_table.end()) return it->second;
      else
      {
          if(use_vg_info)
          {
              KEY inner_key = key;
              int level = compute_level(key);
              for(int i=0; i<level; i++)
              {
                  inner_key >>= 3;
                  it = _hash_table.find(inner_key);
                  if(it != _hash_table.end())
                    return it->second;
              }
          }
          return -1;
      }
  }

  GeneralVoxelGrid<VALUE> to_voxel_grid()
  {
      int resolution = pow(2, _max_level);
      GeneralVoxelGrid<VALUE> ret(resolution, resolution, resolution);

      int counter = 0;
      for(typename std::map<KEY, VALUE>::iterator iter = _hash_table.begin(); iter != _hash_table.end(); ++iter)
      {
          int level = compute_level(iter->first);

          KEY code = (iter->first & ~(KEY(1) << level * 3)) << (_max_level - level) * 3;
          OctreeCoord c;
          c.l = level;
          KEY x, y, z;
          inverse_morton_3d(x, y, z, code);
          c.x = x;
          c.y = y;
          c.z = z;

          int len = int(pow(2, _max_level - level));
          for(int i=0; i < len; i++)
          {
              for(int j=0; j < len; j++)
              {
                  for(int k=0; k < len; k++)
                  {
                      if(iter->second) ret.set_element(i, j, k, iter->second);
                  }
              }
          }
          counter++;
      }

      return ret;
  }

  void from_voxel_grid(GeneralVoxelGrid<VALUE>& vg, int min_level)
  {
      int level = log2((float)vg.depth());
      int dim = vg.depth();

      _max_level = level;

      KEY *keys_arr = new KEY[dim*dim*dim];
      VALUE *values_arr = new VALUE[dim*dim*dim];

      //initially fill the hash map
      for(unsigned int i=0; i<vg.depth(); i++)
      {
          for(unsigned int j=0; j<vg.width(); j++)
          {
              for(unsigned int k=0; k<vg.height(); k++)
              {
                  VALUE val = vg.get_element(i, j, k);
                  OctreeCoord crd;
                  crd.x = i; crd.y = j; crd.z = k; crd.l = level;
                  KEY key = compute_key(crd);
                  keys_arr[i*dim*dim + j*dim + k] = key;
                  values_arr[i*dim*dim + j*dim + k] = val;
              }
          }
      }

      while(level > min_level)
      {
          int step = pow(2, _max_level - level + 1);
          for(unsigned int i=0; i<vg.depth(); i+=step)
          {
              for(unsigned int j=0; j<vg.width(); j+=step)
              {
                  for(unsigned int k=0; k<vg.height(); k+=step)
                  {
                      KEY key = keys_arr[i*dim*dim + j*dim + k];
                      if(compute_level(key) == level)
                      {
                          VALUE comp_val = 0;
                          int el_count = 0;
                          for(int ii=0; ii<2; ii++)
                          {
                              for(int jj=0; jj<2; jj++)
                              {
                                  for(int kk=0; kk<2; kk++)
                                  {
                                      KEY comp_key = keys_arr[(i+ii*step/2)*dim*dim + (j+jj*step/2)*dim + k+kk*step/2];
                                      int comp_lev = compute_level(comp_key);
                                      if(comp_lev == level)
                                      {
                                          VALUE val = values_arr[(i+ii*step/2)*dim*dim + (j+jj*step/2)*dim + k+kk*step/2];
                                          if(val == CLASS_FILLED) comp_val += 1;
                                          el_count += 1;
                                      }
                                  }
                              }
                          }

                          if( (comp_val==0 || comp_val==8) && el_count==8 )
                          {
                              KEY new_key = key >> 3;
                              for(int ii=0; ii<step; ii++)
                              {
                                  for(int jj=0; jj<step; jj++)
                                  {
                                      for(int kk=0; kk<step; kk++)
                                      {
                                          keys_arr[(i+ii)*dim*dim + (j+jj)*dim + k+kk] = new_key;
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }
          level--;
      }

      for(unsigned int i=0; i<vg.depth(); i++)
      {
          for(unsigned int j=0; j<vg.width(); j++)
          {
              for(unsigned int k=0; k<vg.height(); k++)
              {
                  KEY key = keys_arr[i*dim*dim + j*dim + k];
                  _hash_table[key] = values_arr[i*dim*dim + j*dim + k];
              }
          }
      }

      delete[] keys_arr;
      delete[] values_arr;
  }

  void to_file(std::string fname)
  {
      std::ofstream ff(fname.c_str(), std::ios_base::binary);
      boost::archive::text_oarchive oarch(ff);
      std::map<KEY, VALUE> tmp_hash;
      for(typename std::tr1::unordered_map<KEY, VALUE>::iterator it=_hash_table.begin(); it!=_hash_table.end(); it++)
      {
        tmp_hash[it->first] = it->second;
      }
      oarch << tmp_hash;
      ff.flush();
      ff.close();
  }

  void from_file(std::string fname)
  {
      std::ifstream ff(fname.c_str(), std::ios_base::binary);
      boost::archive::text_iarchive iarch(ff);
      std::map<KEY, VALUE> tmp_map;
      iarch >> tmp_map;
      ff.close();

      for(typename std::map<KEY, VALUE>::iterator it=tmp_map.begin(); it!=tmp_map.end(); it++)
      {
        _hash_table.insert(std::pair<KEY, VALUE>(it->first, it->second));
        int level = compute_level(it->first);
        if(level > _max_level) _max_level = level;
      }
  }

};

#endif //OCTREE_H_
