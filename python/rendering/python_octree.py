import math
import numpy as np


def morton3d(x, y, z):
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249

    y = (y | (y << 16)) & 0x030000FF
    y = (y | (y << 8)) & 0x0300F00F
    y = (y | (y << 4)) & 0x030C30C3
    y = (y | (y << 2)) & 0x09249249

    z = (z | (z << 16)) & 0x030000FF
    z = (z | (z << 8)) & 0x0300F00F
    z = (z | (z << 4)) & 0x030C30C3
    z = (z | (z << 2)) & 0x09249249

    return np.uint32(x | (y << 1) | (z << 2))


def inverse_morton3d(z):
    x = z & 0x09249249
    y = (z >> 1) & 0x09249249
    z = (z >> 2) & 0x09249249

    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >> 16) | x) & 0x000003FF

    y = ((y >> 2) | y) & 0x030C30C3
    y = ((y >> 4) | y) & 0x0300F00F
    y = ((y >> 8) | y) & 0x030000FF
    y = ((y >> 16) | y) & 0x000003FF

    z = ((z >> 2) | z) & 0x030C30C3
    z = ((z >> 4) | z) & 0x0300F00F
    z = ((z >> 8) | z) & 0x030000FF
    z = ((z >> 16) | z) & 0x000003FF

    return x, y, z


def max_level():
    return 4 * 8 // 3


def min_level():
    return 0


def clz(x):
    return (bin(x)[2:].zfill(32)+'1').index('1')


def compute_level(key):
    return (max_level() * 3 - clz(key) + 1) // 3


def compute_key(x, y, z, l):
    return morton3d(np.uint32(x), np.uint32(y), np.uint32(z)) \
        | (np.uint32(1) << 3 * l)


def compute_coord(key):
    l = compute_level(key)
    x, y, z = inverse_morton3d(key & ~(np.uint32(1) << l * 3))
    return x, y, z, l


def import_ot(file_name):
    output_model = {}
    max_level = 0
    with open(file_name) as f:
        content = f.readlines()
    parts = content[0].split(" ")
    for i in range(9, len(parts), 2):
        output_model[parts[i]] = parts[i+1]
        lev = compute_level(np.uint32(parts[i]))
        if lev > max_level:
            max_level = lev
    return output_model, pow(2, max_level)


def octree_to_voxel_grid(ot, resolution):
    output = np.zeros((resolution, resolution, resolution))
    max_level = np.uint32(math.log(resolution, 2))

    for key in ot:
        l = compute_level(np.uint32(key))
        code = (np.uint32(key) & ~(np.uint32(1) << l * 3)) \
            << (max_level - l) * 3
        x, y, z = inverse_morton3d(code)
        cube_len = np.uint32(pow(2, max_level - l))
        for i in range(0, cube_len):
            for j in range(0, cube_len):
                for k in range(0, cube_len):
                    if ot[key] == "1":
                        output[x+i, y+j, z+k] = 1

    return output


def get_cube_params(key, resolution):
    l = compute_level(np.uint32(key))
    max_level = np.uint32(math.log(resolution, 2))

    code = (np.uint32(key) & ~(np.uint32(1) << np.uint32(l * 3))) \
        << np.uint32((max_level - l) * 3)

    x, y, z = inverse_morton3d(code)
    side_len = pow(2, max_level - l)

    x = x + float(side_len) / 2.0
    y = y + float(side_len) / 2.0
    z = z + float(side_len) / 2.0

    return x, y, z, side_len


def intersection_over_union(gt, pr):
    arr_int = np.multiply(gt, pr)
    arr_uni = np.add(gt, pr)
    return float(np.count_nonzero(arr_int)) / float(np.count_nonzero(arr_uni))
