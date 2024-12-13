import numpy as np
from scipy.io import loadmat
from os.path import join as pjoin, isdir, realpath, dirname, basename

"""

Details on the origins of these parcellation could be found in 
 loadPrecomputedAtlasCorticalParcellation.m in src repo from Raamana's PhD.
 The internal structure of those Matlab structs contains the precomputed
    parcellation is (generated by Raamana at the tail end of his PhD using script
    subdivideCorticalSurfaceIntoPatchesWithKmeansClustering.m)
 They are located on Cedar HPC (compute Canada) at this path as of 01Apr2020
 /project/6000661/raamana/hpc3194/u1/ADNI/processed/atlases
 /CorticalSubdivisionIntoPatches
    
 Their structure
    >> AdaptivePatchSize
        whole: [1x1 struct]
         left: [1x1 struct]
        right: [1x1 struct]

    >> AdaptivePatchSize.whole
                           tri: [655360x3 int32]
                         coord: [3x327684 double]
              PartitionCenters: [565x3 double]
                PartitionIndex: [1x327684 double]
        NumberedPartitionIndex: [1x327684 double]

    >> AdaptivePatchSize.left
                           tri: [327680x3 int32]
                         coord: [3x163842 double]
                PartitionIndex: [1x163842 double]
        NumberedPartitionIndex: [1x163842 double]

    >> AdaptivePatchSize.right
                           tri: [327680x3 int32]
                         coord: [3x163842 double]
                PartitionIndex: [1x163842 double]
        NumberedPartitionIndex: [1x163842 double]

    this needs about 14-15MB to save on disk in .mat format. However, we need this
    1) in a python native format and
    2) that too only the NumberedPartitionIndex for the whole atlas of fsaverage
    3) to offer this as part of graynet pkg resources, we need it to be smallest
    possible.

    Hence, we will retain only the AdaptivePatchSize.whole.NumberedPartitionIndex
    for now in a numpy array format, that are added to pkg resources, while the
    hefty Matlab struct mat files will be retained in a github repo
    
"""

atlas_name = 'fsaverage'

# min number of vertices per patch
patch_sizes = (250, 500, 1000, 2000, 3000, 5000, 10000)

# base_dir = '/project/6000661/raamana/hpc3194/u1/ADNI/processed/atlases/'
base_dir = '/Users/Reddy/dev/graynet/graynet/resources'
parc_dir = pjoin(base_dir, 'CorticalSubdivisionIntoPatches', atlas_name)

fname = lambda mm: 'AdaptivePatchSize_ParitioningOfCortex_Kmeans_mahalanobis_' \
                   'MinVertexCountPerPartition_{}.mat'.format(mm)

out_name = lambda mm2: 'CortexSubdivision_Kmeans_MinVertexCountPerPartition{}.npy' \
                       ''.format(mm2)

for mvp in patch_sizes:
    raw_raw = loadmat(pjoin(parc_dir, fname(mvp)), matlab_compatible=True)
    raw_struct = raw_raw['AdaptivePatchSize']
    num_part_index = raw_struct[0]['whole'][0]['NumberedPartitionIndex'][0][0]

    np.save(pjoin(parc_dir, out_name(mvp)), num_part_index)

