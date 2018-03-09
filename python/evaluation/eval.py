import numpy as np
import sys

from binvox_rw import binvox_rw as bv

def iou(gt, pr):
    arr_int = np.multiply(gt, pr)
    arr_uni = np.add(gt, pr)
    return float(np.count_nonzero(arr_int)) / float(np.count_nonzero(arr_uni))

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage 'python eval.py <pred model>.binvox <gt model>.binvox'")
		exit(0)

	with open(sys.argv[1]) as f:
		pr_vox = bv.read_as_3d_array(f)

	with open(sys.argv[2]) as f:
		gt_vox = bv.read_as_3d_array(f)

	res = iou(gt_vox.data, pr_vox.data)
	print(round(res, 3))
