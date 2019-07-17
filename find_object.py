# Author @GuruMulay

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io

from skimage.feature import match_descriptors, ORB, plot_matches
from sklearn.cluster import DBSCAN


class Test(object):
    def __init__(self, test_image, orb_file, verbose, testall):
        self.test_image = test_image
        self.orb_file = orb_file
        self.verbose = verbose
        self.testall = testall
        self.test_image_shape = None
        
        # training ORB features 
        self.train_orb_features = np.load(self.orb_file)
        
        self.keypoints_test = None
        self.descriptors_test = None
        self.matches = None
        
        # test set (30 images from the original data)
        self.test_dir = "./test/"
        self.labels_file = os.path.join(self.test_dir, "labels.txt")
        self.test_dict = defaultdict(list)
        
        # predictions
        self.predicted_kprc = []  # (n, 2) for n train image features, stores n predicted keypoint in r, c format
        self.predicted_centroid_rc = np.array([0, 0])  # final predicted r, c location of the object
        
        # clustering params
        self.db_thresh = 25  # 25 pixel DBSCAN clustering threshold
        self.descriptor_extractor = ORB(n_keypoints=50, fast_n=9, fast_threshold=0.15)
        
        # evaluation
        self.pck_at_dot05_thresh = 0.05
    
    def load_test_data(self,):
        if self.verbose: print("loading", self.labels_file)
        with open(self.labels_file, 'r') as l:
            labels = l.readlines()
        for li, row in enumerate(labels):
            l = row.split(' ')
            self.test_dict[l[0]].append(float(l[1]))
            self.test_dict[l[0]].append(float(l[2]))
    
    
    def print_kp_d(self, image_text, keypoints, descriptors):
        print(image_text)
        print("keypoints r, c", keypoints.shape)
        print("descriptors", descriptors.shape)
       
    
    def isWithinNormalizedThreshold(self, a, b):
        """
        check for image if predicted keypoint (b) is within thresh distance of ground truth keypoint (a)
        """
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5 < self.pck_at_dot05_thresh
    
    
    def cluster_keypoints_get_centroid(self, nkps):
        """
        given n keypoints (n, 2), find clusters, return the centroid (2,) of the dominant (main) cluster 
        """
        db = DBSCAN(eps=self.db_thresh, min_samples=5).fit(nkps)
        labels = db.labels_
        ld = defaultdict(int)  # stores cluster_id: number of candidates in that cluster 
        contains_noise_points = False
        for l in labels:
            ld[l] += 1
            if l == -1: contains_noise_points = True  # make a note that noise points are present (to remove them later while chosing largest cluster)
        if contains_noise_points: ld.pop(-1)  # remove if present the noise points cluster from dict
            
        if bool(ld):  # if there are clusters
            # sort the cluster dict to chose largest cluster (sort because we will chose top cluster 
            # if there are more than one clusters that have maximum candidates)
            ld_sorted = sorted(ld.items(), key=lambda kv: kv[1], reverse=True)  
            max_cluster_id = ld_sorted[0][0]
            # print('Estimated number of clusters:', len(set(labels)) - (1 if -1 in labels else 0))
            # print('Estimated number of noise points:', list(labels).count(-1))
            ids_main_cluster = np.where(labels==max_cluster_id)  # ids of dominant (main) cluster candidates
            kps_main_clustered = nkps[ids_main_cluster]  # kps of dominant (main) cluster candidates
            mean_predicted_cluster = np.mean(kps_main_clustered, axis = 0)
            return mean_predicted_cluster
        else:
            return np.array([False, False])
    
    
    def test_object_detector_single_image(self,):
        if self.verbose: print("loading ", self.orb_file, self.train_orb_features.shape, " testing ", self.test_image)
        testimg = rgb2gray(skimage.io.imread(self.test_image))
        self.test_image_shape = testimg.shape
        # ORB keypoints and descriptors
        self.descriptor_extractor.detect_and_extract(testimg)
        self.keypoints_test = self.descriptor_extractor.keypoints
        self.descriptors_test = self.descriptor_extractor.descriptors
        
        # match the test features against the stored train ORB features
        for i in range(self.train_orb_features.shape[0]):  # for n train image features
            # print("self.train_orb_features[i]", self.train_orb_features[i].shape)
            self.matches = match_descriptors(self.train_orb_features[i], self.descriptors_test, cross_check=True)
            # print("self.matches", self.matches, self.matches.shape)  # (_, 2)
            
            # cluster the matched keypoints and chose the cluster with highest number of matches
            kp_ids = self.matches[:, 1]
            object_kps = self.keypoints_test[kp_ids]  # (_, 2)            
            if self.cluster_keypoints_get_centroid(object_kps).any():
                self.predicted_kprc.append(self.cluster_keypoints_get_centroid(object_kps))
        
        self.predicted_kprc = np.asarray(self.predicted_kprc)
        # print("final predicted object locations r, c \n", self.predicted_kprc, self.predicted_kprc.shape)
        
        # cluster the n predictions and use the centroid of the strongest cluster as the final prediction
        self.predicted_centroid_rc = self.cluster_keypoints_get_centroid(self.predicted_kprc)
        if self.verbose: print("predicted output in (c, r) i.e., (x, y) format")
        print(round(self.predicted_centroid_rc[1]/self.test_image_shape[1], 4), round(self.predicted_centroid_rc[0]/self.test_image_shape[0], 4))
        self.predicted_kprc = []
        
    
    def test_and_evaluate_testset(self,):
        self.load_test_data()
        n_correct = 0
        # for test images in ./test folder
        for imgid, objectcr in self.test_dict.items():
            print("img and ground truth ============", imgid, objectcr)
            self.test_image = os.path.join(self.test_dir, imgid)
            self.test_object_detector_single_image()
            if self.isWithinNormalizedThreshold(objectcr, [self.predicted_centroid_rc[1]/self.test_image_shape[1], self.predicted_centroid_rc[0]/self.test_image_shape[0]]):
                n_correct += 1
            
        print("==============================================================")
        print("Accuracy: Percentage of Correct Keypoint (PCK@0.05) @normalized_distance of 0.05 == {} %".format( round(100*n_correct/len(self.test_dict.keys()), 2)))
        print("==============================================================")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finds the location of object by loading ORB features saved during training and "
                                     " performing keypoint matching.")

    parser.add_argument('imagefile', help='path to test image')
    parser.add_argument('-orb', '--orbdescriptor', help='path to ORB descriptor .npy file; default is ./train_descriptors.npy', default="./train_descriptors.npy")
    parser.add_argument('-v', '--verbose', help='verbose output with detailed results', action="store_true", default=False)
    parser.add_argument('-t', '--testall', help='test all the images in ./test directory ignoring the input image', action="store_true", default=False)
    args = parser.parse_args()

    t = Test(test_image=args.imagefile, orb_file=args.orbdescriptor, verbose=args.verbose, testall=args.testall)
    if not args.testall:
        t.test_object_detector_single_image()
    else:
        t.test_and_evaluate_testset()  # evaluate test set from ./test folder