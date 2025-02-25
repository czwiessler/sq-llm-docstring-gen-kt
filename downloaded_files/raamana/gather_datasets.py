import pickle
from os.path import exists as pexists, join as pjoin

import networkx as nx
import numpy as np
from neuropredict import run_workflow
from pyradigm import MLDataset

from graynet import parcellate

"""#---------------------------------#---------------------------------#---------------------------------

You can use this script to traverse through the ouputs generated, and know which ones failed and need to be resubmitted.

Info regarding 

This script needs to be modified to suit your needs. It can  also produce a nice pyradigm dataset for use later on with neuropredict, if you are interested in predictive modeling.

"""#---------------------------------#---------------------------------#---------------------------------


#---------------------------------
# CHANGE THESE FOR YOUR PROCESSING
#---------------------------------

base_dir = '/u1/work/hpc3194'
dataset_list = ['PPMI', '4RTNI']

# freesurfer_dir = pjoin(proc_dir, 'freesurfer')
# subject_id_list = pjoin(target_list_dir, 'graynet.compute.list')

numeric_labels = {'CN' : 1, 'PARK' : 2, 'CBS' : 3, 'PSP': 4}

base_feature_list  = ['freesurfer_thickness', 'freesurfer_curv', 'freesurfer_sulc', 'freesurfer_area']
multi_feature_list = ['freesurfer_thickness_curv_sulc_area', ]
edge_range = {'freesurfer_thickness': (0.0, 5.0),
              'freesurfer_curv'     : (-0.3, +0.3),
              'freesurfer_sulc'     : (-1.5, +1.5),
              'freesurfer_area'     : (0.0, 1.5)
              }
expt_prefix = 'thk_curv_sulc_area_nbins25'
file_ext = 'graynet.graphml'
atlas = 'fsaverage'  # 'glasser2016' # 'fsaverage' # 'glasser2016' #
fwhm = 10
node_size = None

num_rois = {'glasser2016': 360, 'fsaverage': 68}
num_links_expected = num_rois[atlas]*(num_rois[atlas]-1)/2

num_bins = 25

#---------------------------------
# END MAKING CHANGES
#---------------------------------


histogram_dist = np.array(['chebyshev', 'chi_square', 'correlate', 'cosine', 'euclidean',
                           'histogram_intersection', 'jensen_shannon', 'manhattan', 'minowski',  'relative_deviation'])

atlas_rois, centroids, vertex_labels = parcellate.roi_labels_centroids(atlas)

def get_weights_order(graph, nodes=atlas_rois):
    "returns weights in the order of nodes requested"

    # order is not guaranteed below
    edge_dict = nx.get_edge_attributes(graph, 'weight')
    # so ordering it here, to ensure correspondence across subjects
    weights = [ graph[x][y]['weight'] for x in nodes for y in nodes if (x,y) in edge_dict ]

    return np.array(weights)


incomplete_processing, comb_nan_values = dict(), dict()
for base_feature in base_feature_list:
    incomplete_processing[base_feature] = dict()
    comb_nan_values[base_feature] = dict()

    for weight_method in histogram_dist: # ['chi_square', 'cosine', 'euclidean', 'histogram_intersection']:
        # print('Gathering data for {}'.format(weight_method))
        # freesurfer_area_fsaverage_smoothing10_sizeNone_edgeweight_chi_square_graynet.graphml
        expt_id = '{}_{}_smoothing{}_size{}_edgeweight_{}'.format(base_feature, atlas, fwhm, node_size,
                                                                  weight_method)

        flag_nan_exists = False
        flag_incomplete = False
        flag_unexpected = False
        dataset = MLDataset()

        incomplete_processing[base_feature][weight_method] = dict()
        comb_nan_values[base_feature][weight_method] = dict()
        for ds_name in dataset_list:
            proc_dir = pjoin(base_dir, ds_name, 'processed')
            # out_dir = pjoin(proc_dir, 'graynet', '{}_{}_fwhm{}'.format(base_feature, atlas, fwhm))
            out_dir = pjoin(proc_dir, 'graynet', '{}_{}_fwhm{}'.format(expt_prefix, atlas, fwhm))

            meta_list = pjoin(proc_dir, 'target_lists', 'meta_{}.csv'.format(ds_name))
            sample_ids, classes = run_workflow.get_metadata(meta_list)

            incomplete_processing[base_feature][weight_method][ds_name] = list()
            comb_nan_values[base_feature][weight_method][ds_name] = list()
            for sample in sample_ids:
                feat_path = pjoin(out_dir, sample, '{}_{}'.format(expt_id, file_ext))
                if pexists(feat_path):
                    graph = nx.read_graphml(feat_path)
                    data = get_weights_order(graph, atlas_rois)
                    idx_nan = np.logical_not(np.isfinite(data))
                    local_flag_nan_exists = np.count_nonzero(idx_nan) > 0
                    if local_flag_nan_exists:
                        flag_nan_exists = True
                        comb_nan_values[base_feature][weight_method][ds_name].append(sample)
                        # print('NaNs found for {} {} {}'.format(ds_name, weight_method, sample))
                    elif len(data) >= num_links_expected:
                        dataset.add_sample(sample, data, numeric_labels[classes[sample]], class_id=classes[sample])
                    else:
                        flag_unexpected = True
                        incomplete_processing[base_feature][weight_method][ds_name].append(sample)
                else:
                    flag_incomplete = True
                    incomplete_processing[base_feature][weight_method][ds_name].append(sample)
                    # print('processing incomplete for {} {} {}'.format(ds_name, weight_method, sample))

        if flag_nan_exists or flag_incomplete or flag_unexpected:
            pass
            # print('{:20} {:25} - processing unusable; totally skipping it.'.format(base_feature, weight_method))
        else:
            print('{:20} {:5} {:25} - fully usable.'.format(base_feature, ds_name, weight_method))
            dataset.description = '{}_{}'.format(base_feature, weight_method)
            out_path = pjoin(out_dir, '{}_{}.MLDataset.pkl'.format(base_feature, weight_method))
            dataset.save(out_path)

    # saving
    with open(pjoin(out_dir, 'incomplete_unusable_processing.pkl'), 'wb') as ipf:
        pickle.dump([incomplete_processing, comb_nan_values], ipf)

# reading
with open(pjoin(out_dir, 'incomplete_unusable_processing.pkl'), 'rb') as ipf:
    incomplete_processing, comb_nan_values = pickle.load(ipf)

# results
for base_feature in base_feature_list:
    for ds_name in dataset_list:
        for weight_method in histogram_dist:
            print('{:20} {:5} {:25} {:5} {:5}'.format(base_feature, ds_name, weight_method,
                len(incomplete_processing[base_feature][weight_method][ds_name]),
                len(comb_nan_values[base_feature][weight_method][ds_name])))
