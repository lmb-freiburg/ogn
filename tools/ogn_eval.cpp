#include <boost/program_options.hpp>
#include <iostream>

#include "image_tree_tools/image_tree_tools.h"

std::string prediction_file, reference_file;

int register_cmd_options(int argc, char* argv[]) {
    try {
        boost::program_options::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("prediction,p", boost::program_options::value<std::string>(&prediction_file)->required(), "Predicted model file")
            ("reference,r", boost::program_options::value<std::string>(&reference_file)->required(), "Reference model file")
        ;

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if ( vm.count("help") ) {
            std::cout << desc << std::endl;
        } else if ( !vm.count("prediction") || !vm.count("reference") ) {
            std::cout << desc << std::endl;
            return -1;
        }
        
        prediction_file = vm["prediction"].as<std::string>();
        reference_file = vm["reference"].as<std::string>();
    } catch( boost::program_options::required_option& e ) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return -1;
    }
    return 0;
}

float iou(VoxelGrid ref, VoxelGrid pr)
{
	float cnt_i = 0, cnt_u = 0;
	for(int i=0; i<ref.depth(); i++)
	{
		for(int j=0; j<ref.width(); j++)
		{
			for(int k=0; k<ref.height(); k++)
			{
				cnt_i += ref.get_element(i, j, k) && pr.get_element(i, j, k);
				cnt_u += ref.get_element(i, j, k) || pr.get_element(i, j, k);
			}
		}
	}
	return cnt_i / cnt_u;
}

int main(int argc, char* argv[]) {
    if ( !register_cmd_options(argc, argv) ) {
    	VoxelGrid vg_ref, vg_pred;

        std::string prediction_ext = get_file_extension(prediction_file);
        std::string reference_ext = get_file_extension(reference_file);

        if ( prediction_ext == "ot" ) {
        	Octree octree;
            octree.from_file(prediction_file);
            vg_pred = octree.to_voxel_grid();
        } else if ( prediction_ext == "binvox" ) {
            vg_pred.read_binvox(prediction_file);
        }

        if ( reference_ext == "ot" ) {
        	Octree octree;
            octree.from_file(reference_file);
            vg_ref = octree.to_voxel_grid();
        } else if ( reference_ext == "binvox" ) {
            vg_ref.read_binvox(reference_file);
        }

        std::cout << iou(vg_ref, vg_pred) << std::endl;

        return 0;
	}
}