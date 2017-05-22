#ifndef IMAGE_TREE_TOOLS_H_
#define IMAGE_TREE_TOOLS_H_

#include "zindex.h"
#include "voxel_grid.h"
#include "octree.h"
#include "common_util.h"

#define CLASS_MIXED 2
#define CLASS_IGNORE 3

#define PROP_FALSE 0
#define PROP_TRUE 1

#define OGN_NUM_CLASSES 3

typedef unsigned int KeyType;
typedef byte SignalType;
typedef OccupancyVoxelGrid VoxelGrid;
typedef GeneralOctree<SignalType> Octree;

#endif //IMAGE_TREE_TOOLS_H_
