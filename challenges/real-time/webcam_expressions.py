from pathlib import Path

import torch
from torchvision import transforms
import albumentations as albu 

import tracking as trk
from tracking import Image
from tracking.detect import FaceMTCNN
from tracking.annotate import load_osnet
import time

import pytorchfer
import cv2 
import numpy as np 
import torch.hub 
from pytorchfer.src import model
from torchsummary import summary
from pytorchfer.src.visualize.grad_cam import BackPropagation, GradCAM, GuidedBackPropagation

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

    def expression(self, frame, objects, index):
        # image augmentation helper function
        def augment(aug, image):
            return aug(image=image)['image']
        # convert box float vals to ints for cropping faces
        box_ints = objects[index]['box'].astype(int)
        # crop bounding box, resize to 48x48, convert to 1-channel tensor
        crop_aug = albu.Crop(
            box_ints[0], box_ints[1],
            box_ints[2], box_ints[3])
        cropped = augment(crop_aug, frame)
        size_aug = albu.Resize(48, 48)
        cropped = augment(size_aug, cropped)
        pil = transforms.ToPILImage()(cropped)
        grayed = transforms.Grayscale()(pil)
        input_tens = transforms.ToTensor()(grayed)

        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
        # define and load network
        net = model.Model(num_classes=len(classes))
        checkpoint = torch.load('../pytorchfer/trained/private_model_291_60.t7', 
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        # backprop net - order of predicted classes
        img = torch.stack([input_tens])
        bp = BackPropagation(model=net)
        prob, emo = bp.forward(img)
        gcam = GradCAM(model=net)
        _ = gcam.forward(img)
        gbp = GuidedBackPropagation(model=net)
        _ = gbp.forward(img)
        # most probable class
        actual_emotion = emo[:,0]
        gbp.backward(ids=actual_emotion.reshape(1,1))
        gcam.backward(ids=actual_emotion.reshape(1,1))

        print(classes[actual_emotion.data], (prob.data[:,0] * 100))     


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
            self.expression(frame, objs, n)
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

    # visualize boxes
    image = visualizer(image)
    window.write(image)
    prev_frame_time = time.time()
window.close_all()
cam.close()
