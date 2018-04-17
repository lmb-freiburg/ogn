#ifndef VOXEL_GRID_H
#define VOXEL_GRID_H

#include <boost/shared_array.hpp>

#include <string>
#include <fstream>
#include <iostream>

//OCCUPANCY SIGNAL VALUES
#define CLASS_EMPTY 0
#define CLASS_FILLED 1

typedef unsigned char byte;

template <class VALUE>
class GeneralVoxelGrid
{

protected:
    int _depth, _height, _width;
    boost::shared_array<VALUE> _voxels;

public:
    GeneralVoxelGrid()
    {
        _depth = 0; _height = 0; _width = 0;
        _voxels = boost::shared_array<VALUE>();
    }

    GeneralVoxelGrid(int depth, int height, int width)
    {
        _depth = depth; _height = height; _width = width;
        _voxels = boost::shared_array<VALUE>(new VALUE[_depth * _height * _width]);
        memset(_voxels.get(), VALUE(0), _depth * _height * _width);
    }

    int size() {return _depth * _height * _width;}

    int depth() {return _depth;}

    int width() {return _width;}

    int height() {return _height;}

    VALUE get_element(int i, int j, int k) {return _voxels[i * _width * _height + j * _height + k];}
    void set_element(int i, int j, int k, VALUE val) {_voxels[i * _width * _height + j * _height + k] = val;}
};

class OccupancyVoxelGrid : public GeneralVoxelGrid<byte>
{

public:

  OccupancyVoxelGrid()
    : GeneralVoxelGrid(){}
  OccupancyVoxelGrid(int depth, int height, int width)
    : GeneralVoxelGrid(depth, height, width){}

  int write_binvox(std::string filespec)
  {
      std::ofstream output(filespec.c_str(), std::ios::binary);
      output << "#binvox 1\n";
      output << "dim " << _depth << " " << _height << " " << _width << "\n";
      output << "translate 0 0 0\n";
      output << "scale 1\n";
      output << "data\n";

      byte count = 0;
      byte prev_value = _voxels[0];
      for(int i=1; i<size(); i++)
      {
          count++;
          if(_voxels[i] != prev_value || count == 255)
          {
              output << prev_value;
              output << count;

              prev_value = _voxels[i];
              count = 0;
          }
      }
      
      output.close();
      return 0;
  }

  int read_binvox(std::string filespec)
  {
      std::ifstream *input = new std::ifstream(filespec.c_str(), std::ios::in | std::ios::binary);

      // read header
      std::string line;
      std::string version;
      *input >> line;  // #binvox
      if (line.compare("#binvox") != 0) {
        std::cout << "Error: first line reads [" << line << "] instead of [#binvox]" << std::endl;
        delete input;
        return 0;
      }
      *input >> version;

      _depth = -1;
      int done = 0;
      while(input->good() && !done) {
        *input >> line;
        if (line.compare("data") == 0) done = 1;
        else if (line.compare("dim") == 0) {
          *input >> _depth >> _height >> _width;
        }
        else {
          char c;
          do {  // skip until end of line
            c = input->get();
          } while(input->good() && (c != '\n'));

        }
      }
      if (!done) {
        std::cout << "  error reading header" << std::endl;
        return 0;
      }
      if (_depth == -1) {
        std::cout << "  missing dimensions in header" << std::endl;
        return 0;
      }

      _voxels = boost::shared_array<byte>(new byte[this->size()]);
      if (!_voxels) {
        std::cout << "  error allocating memory" << std::endl;
        return 0;
      }
      // read voxel data
      byte value;
      byte count;
      int index = 0;
      int end_index = 0;
      int nr_voxels = 0;

      input->unsetf(std::ios::skipws);  // need to read every byte now (!)
      *input >> value;  // read the linefeed char

      while((end_index < size()) && input->good()) {
        *input >> value >> count;

        if (input->good()) {
          end_index = index + count;
          if (end_index > size()) return 0;
          for(int i=index; i < end_index; i++)
          {
              if (value) _voxels[i] = CLASS_FILLED;
              else _voxels[i] = CLASS_EMPTY;
          }

          if (value) nr_voxels += count;
          index = end_index;
        }  // if file still ok

      }  // while
      input->close();
      delete input;
      return 1;
  }
};

#endif // VOXEL_GRID_H
