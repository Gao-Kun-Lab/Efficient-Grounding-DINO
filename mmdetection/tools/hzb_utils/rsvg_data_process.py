import os
import os.path as osp
import pickle

from joblib import load
import torch
import cv2
def process_rsvg_data(data_dir):
    with open(data_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(',')
            img_name = line[0]
    return



if __name__=="__main__":
    data_dir = '/data1/detection_data/rsvg/RSVG-HR/rsvg_hr_train.txt'
    process_rsvg_data(data_dir)