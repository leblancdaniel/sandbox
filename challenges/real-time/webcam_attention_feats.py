from pathlib import Path
import numpy as np 
import math

import torch

import tracking as trk
from tracking import Image
from tracking.detect import FaceMTCNN
from tracking.annotate import load_osnet

import time

cam = trk.io.CamReader(width=1280, height=720, mirror=True)
window = trk.io.WindowWriter("Face Detection Window")
###################################################
class FaceDetector:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # change filepaths for demo
        self.detection = FaceMTCNN(
            "/Users/danielleblanc/cameralytics/models",
            device=device 
        )
        self.features = load_osnet(
            "/Users/danielleblanc/cameralytics/models/osnet075.pth", arch="x0_75"
        )
        self.expression = 

    def __call__(self, frame: Image):
        frame = self.detection(frame)
        # No faces detected, return False
        if len(frame.objects) < 1:
            print("-----------------------NO FACES DETECTED--------------------")
            return frame
        frame = self.features(frame)
        return frame

detector = FaceDetector()
############################################
class FeatureTracker:
    def __init__(self):
        self.id_num = 1
        self.age = 0
        self.age_dict = {}
        self.reset = 10         # replace "noise" every X frames
        self.cos_dict = {}
        self.collection = {}
        self.threshold = 0.85   # Threshold for cosine similarity

    def cossim(self, objects, index):
        for k in self.collection.keys():
            input1 = self.collection[k]
            input2 = objects[index]['features']
            cos = torch.nn.CosineSimilarity(dim=0)
            output = cos(input1, input2)
            self.cos_dict.update({k : output})

    def update(self, objects, index):
        maximum = max(self.cos_dict, key=self.cos_dict.get)
        if self.cos_dict[maximum] >= self.threshold:
            self.collection.update({maximum : objects[index]['features']})
            self.age_dict.setdefault(maximum, []).append(self.age)
            objects[index].update({'id': maximum})
        else:
            self.collection.update({self.id_num : objects[index]['features']})
            self.age_dict.setdefault(self.id_num, []).append(self.age)
            objects[index].update({'id': self.id_num})
            self.id_num += 1 

    def attention(self, image, objects, index):
        # define facial landmark points
        left_eye, right_eye, nose, left_mouth, right_mouth = objects[index]['landmarks'][index]
        mid_eye = (right_eye + left_eye) / 2
        mid_mouth = (right_mouth + left_mouth) / 2
        # define distance from nose base to nose tip & halfway from eye to eye
        face2nose = np.cross(mid_eye - mid_mouth, mid_mouth - nose) / np.linalg.norm(mid_eye - mid_mouth)
        mideye2eye = math.hypot(right_eye[0] - mid_eye[0], right_eye[1] - mid_eye[1])
        # define "attention window"        
        X_ref = mid_eye[0] / image.shape[1]
        attn_rbound = mideye2eye * (X_ref - 1)
        attn_lbound = attn_rbound + mideye2eye
        # if nose length within "attention window" then looking towards the camera
        if face2nose > attn_rbound and face2nose < attn_lbound:
            objects[index].update({'attention': 1})
        else:
            objects[index].update({'attention': 0})

    def denoise(self):
        if self.age == self.reset:   # every X frames remove "noise"
            for k,v in self.age_dict.items():               
                if len(v) <= 1:  # considered "noise" if ID only appears Y times in X frames       
                    self.collection.pop(k, None)
                    self.cos_dict.pop(k, None)
            self.age_dict.clear()
            self.age -= self.reset

    def __call__(self, frame: Image):
        objs = frame.objects
        # if objects detected and ID-feature dictionary empty,
        # update empty dictionary w/ ID-feature pair, and append ID to frame.objects
        self.age += 1
        for n in range(len(objs)):
            if len(self.collection) < 1:
                self.collection.update({n + 1: objs[n]['features']})
                self.age_dict.setdefault(n + 1, []).append(self.age)
                objs[n].update({'id': n + 1})
                self.id_num += 1
                continue
            # if ID-feature dictionary is not empty and face is detected, 
            # calculate cosine similarity to each feature in ID-feature dictionary
            self.cossim(objs, n)
            # if the max cosine similarity of all id features < threshold, 
            # add new ID-feature pair and add new ID number to frame.object          
            self.update(objs, n)
            self.attention(frame, objs, n)
        self.denoise()

        return frame

tracker = FeatureTracker()

# Visualizer takes the object metadata and draws it to the frame
visualizer = trk.visualize.Visualizer()

prev_frame_time = time.time()
for image in cam:
    # Detect faces in camera
    image = detector(image)
    # ID faces based on cosine similarity to previously detected face features
    image = tracker(image)
    print(image.objects)
    # visualize boxes
    image = visualizer(image)
    window.write(image)
    prev_frame_time = time.time()
window.close_all()
cam.close()
