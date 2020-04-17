from pathlib import Path

import torch
from torchvision import transforms
import albumentations as albu 

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
        self.attn_dict = {}
        self.error_time = 5
        self.attn_age = 0
        self.attn_age_dict = {}

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

    def attention(self, frame, objects, index):
        # image augmentation helper function
        def augment(aug, image):
            return aug(image=image)['image']
        # convert box float vals to ints for cropping faces
        box_ints = objects[index]['box'].astype(int)
        box_half_width = (box_ints[2] - box_ints[0]) // 2
        box_mid_width = (box_ints[0] + box_ints[2]) // 2
        # crop only bounding box
        crop_left = albu.Crop(
            (box_mid_width - box_half_width), box_ints[1], 
            box_mid_width, box_ints[3])
        crop_right = albu.Crop(
            box_mid_width, box_ints[1],
            (box_mid_width + box_half_width), box_ints[3])
        cropped_l = augment(crop_left, frame)
        cropped_r = augment(crop_right, frame)
        # increase brightness and decrease contrast for lessened lighting effect
        lighting_aug = albu.RandomBrightnessContrast(
            brightness_limit=(0.3, 0.299), 
            contrast_limit=(-0.2, -0.199), p=1)
        cropped_l = augment(lighting_aug, cropped_l)
        cropped_r = augment(lighting_aug, cropped_r)
        # flip left side of cropped image horizontally
        flip_aug = albu.HorizontalFlip(p=1)
        flipped_l = augment(flip_aug, cropped_l) 
        # convert to tensor and flatten
        cropped_r = torch.flatten(transforms.ToTensor()(cropped_r))
        flipped_l = torch.flatten(transforms.ToTensor()(flipped_l))
        # compare right side of face with flipped left side of face
        cos = torch.nn.CosineSimilarity(dim=0)
        output = cos(cropped_r, flipped_l)
        if output < 0.985:
            attn = 0
        else:
            attn = 1
        # create dictionary of attention flags for each ID
        self.attn_age += 1
        self.attn_dict.setdefault(objects[index]['id'], []).append(attn)
        self.attn_age_dict.update({objects[index]['id']: self.attn_age})

        for v in self.attn_dict.values():
            if len(v) > self.error_time:    # history of X frames per ID
                v.pop(0)                    # remove oldest attention flag for ID
            if len(v) > 2:                  # record mode of last X frames
                objects[index].update({'forward_gaze': max(set(v), key=v.count)})
            else:                           # record current flag
                objects[index].update({'forward_gaze': attn})

        for k,v in self.attn_age_dict.items():
            if (self.attn_age - v) >= 10:
                self.attn_dict.pop(k, None)
        if self.attn_age % 30 == 0:
            self.attn_age_dict.clear()

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
