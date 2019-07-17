# Author @GuruMulay

import os
import argparse
import numpy as np

from collections import defaultdict

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io

from skimage.feature import match_descriptors, ORB, plot_matches


class Train(object):
    def __init__(self, train_dir, verbose):
        self.train_dir = train_dir
        self.verbose = verbose

        self.labels_file = os.path.join(self.train_dir, "labels.txt")
        self.train_on_full_data = True
        self.n_train = 100
        self.train_dict = defaultdict(list)
        
        self.thresh = 25  # 25 pixel threshold to decide if the detected ORB keypoint is within this threshold of the ground truth object location 
        
        self.descriptor_extractor = ORB(n_keypoints=50, fast_n=9, fast_threshold=0.15)
        self.all_train_descriptors = []
        self.train_descriptor_outfile = os.path.join(os.getcwd(), "train_descriptors.npy")
    
    
    def load_train_data(self,):
        if self.verbose: print("loading", self.labels_file)
        with open(self.labels_file, 'r') as l:
            labels = l.readlines()

        for li, row in enumerate(labels):
            if not self.train_on_full_data and li >= self.n_train:
                break
            l = row.split(' ')
            self.train_dict[l[0]].append(float(l[1]))
            self.train_dict[l[0]].append(float(l[2]))
        # print("train_dict", self.train_dict, len(self.train_dict.keys()))


    def isWithinGT(self, a, b):
        """
        check for image if given gt_kprc_object keypoint (a) is within self.thresh distance of detected ORB keypoint (b)
        """
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5 < self.thresh
    
    
    def print_kp_d(self, image_text, keypoints, descriptors):
        print(image_text)
        print("keypoints r, c", keypoints.shape)
        print("descriptors", descriptors.shape)
    
    
    def extract_and_store_ORB_features(self,): 
        total_descriptors = 0
        for imgid, objectcr in self.train_dict.items():
            # print("img ============", imgid, objectcr)
            img = rgb2gray(skimage.io.imread(os.path.join(self.train_dir, imgid)))
            gt_kprc_object = [img.shape[0]*objectcr[1], img.shape[1]*objectcr[0]]  # print("gt_kprc_object", gt_kprc_object)

            # ORB keypoints and descriptors
            self.descriptor_extractor.detect_and_extract(img)
            keypoints_train = self.descriptor_extractor.keypoints
            descriptors_train = self.descriptor_extractor.descriptors
            # if self.verbose: self.print_kp_d("image kp before filter", keypoints_train, descriptors_train)

            # filter detected kp by considering only the ones near to object
            gt_kp_ids = []
            for i, kp in enumerate(keypoints_train):
                if self.isWithinGT(gt_kprc_object, kp):
                    gt_kp_ids.append(i)
            
            keypoints_train = keypoints_train[gt_kp_ids]
            descriptors_train = descriptors_train[gt_kp_ids]  # print("gt_kp_ids", gt_kp_ids, len(gt_kp_ids))
            # if self.verbose: self.print_kp_d("image kp after filter", keypoints_train, descriptors_train)
            total_descriptors += descriptors_train.shape[0]
            
            self.all_train_descriptors.append(descriptors_train)
            
        self.all_train_descriptors = np.asarray(self.all_train_descriptors)   
        if self.verbose: print("total_descriptors, self.all_train_descriptors.shape", total_descriptors, self.all_train_descriptors.shape)
        
        # save all the training descriptors (ORB) to disk in .npy format
        np.save(self.train_descriptor_outfile, self.all_train_descriptors)
        if self.verbose: print("Training complete. Saved all the training descriptors (ORB) to {}".format(self.train_descriptor_outfile))
            

    def train_object_detector(self,):
        self.load_train_data()
        self.extract_and_store_ORB_features()        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm to find the location of object. This script trains on the given input data."
                                     " Stores extracted ORB features that will be later used while testing.")

    parser.add_argument('traindir', help='path to train data dir', nargs='?', default="./train")
    parser.add_argument('-v', '--verbose', help='verbose output with detailed results', action="store_true", default=False)
    args = parser.parse_args()

    t = Train(train_dir=args.traindir, verbose=args.verbose)
    t.train_object_detector()