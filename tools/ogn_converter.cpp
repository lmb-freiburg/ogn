#include <boost/program_options.hpp>
#include <iostream>

#include "image_tree_tools/image_tree_tools.h"

using namespace boost::program_options;
using namespace std;

string input_file, output_file;
int min_level = 0;

int register_cmd_options(int argc, char* argv[])
{
    try
    {
        options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i", value<string>(&input_file)->required(), "Input file name for conversion")
            ("output,o", value<string>(&output_file)->required(), "Output file name for conversion")
            ("min_level,l", value<int>(&min_level), "Minimum octree level")
        ;

        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) cout << desc << endl;

        input_file = vm["input"].as<string>();
        output_file = vm["output"].as<string>();
        min_level = vm["min_level"].as<int>();
    }
    catch(boost::program_options::required_option& e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    if(!register_cmd_options(argc, argv))
    {
        Octree octree;

        cout << "Input file: " << input_file << endl;
        cout << "Output file: " << output_file << endl;
        cout << "Minimum level: " << min_level << endl;

        string input_ext = split(input_file, '.')[1];
        string output_ext = split(output_file, '.')[1];

        //read converter input
        if(input_ext == "ot")
        {
            octree.from_file(input_file);
        }
        else if(input_ext == "binvox")
        {
            VoxelGrid vg;
            vg.read_binvox(input_file);
            octree.from_voxel_grid(vg, min_level);
        }


        //generate converter output
        if(output_ext == "ot")
        {
            octree.to_file(output_file);
        }
        else if(output_ext == "binvox")
        {
            //VoxelGrid vg = octree.to_voxel_grid();
            //vg.write_binvox(output_string);
        }
    }

    return 0;
}
