#include <boost/program_options.hpp>
#include <iostream>

#include "image_tree_tools/image_tree_tools.h"

std::string input_file, output_file;
int min_level = 0;

int register_cmd_options(int argc, char* argv[]) {
    try {
        boost::program_options::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i", boost::program_options::value<std::string>(&input_file)->required(), "Input file name for conversion")
            ("output,o", boost::program_options::value<std::string>(&output_file)->required(), "Output file name for conversion")
            ("min_level,l", boost::program_options::value<int>(&min_level), "Minimum octree level")
        ;

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if ( vm.count("help") ) {
            std::cout << desc << std::endl;
        } else if ( !vm.count("input") || !vm.count("output") || !vm.count("min_level") ) {
            std::cout << desc << std::endl;
            return -1;
        }
        
        input_file = vm["input"].as<std::string>();
        output_file = vm["output"].as<std::string>();
        min_level = vm["min_level"].as<int>();
    } catch( boost::program_options::required_option& e ) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if ( !register_cmd_options(argc, argv) ) {
        Octree octree;

        std::cout << "Input file: " << input_file << std::endl;
        std::cout << "Output file: " << output_file << std::endl;
        std::cout << "Minimum level: " << min_level << std::endl;

        std::string input_ext = get_file_extension(input_file);
        std::string output_ext = get_file_extension(output_file);

        //read converter input
        if ( input_ext == "ot" ) {
            octree.from_file(input_file);
        } else if ( input_ext == "binvox" ) {
            VoxelGrid vg;
            vg.read_binvox(input_file);
            octree.from_voxel_grid(vg, min_level);
        }


        //generate converter output
        if ( output_ext == "ot" ) {
            octree.to_file(output_file);
        } else if ( output_ext == "binvox" ) {
            //VoxelGrid vg = octree.to_voxel_grid();
            //vg.write_binvox(output_string);
        }
    }
    return 0;
}
