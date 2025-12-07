import os
import os.path as osp
import pickle
import json

def rrsisd_data(data_dir):
    ref_file = osp.join(data_dir,'rrsisd/refs(unc).p')
    instances_file = osp.join(data_dir, 'rrsisd/instances.json')
    ref_data = pickle.load(open(ref_file, 'rb'))
    instances_data = json.load(open(instances_file, 'r'))
    return


if __name__=='__main__':
    data_dir = '/data1/detection_data/rrsisd/'
    rrsisd_data(data_dir)