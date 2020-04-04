import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2

import sys
import argparse

import time
from PIL import Image
from matplotlib import pyplot as plt

import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", dest="model_path", help="Path to the pth file of the pretrained model",
                    type=str)

parser.add_argument("--model_type", dest="model_type", help="Currently support : resnet18 | mobilenet",
                    type=str)

args = parser.parse_args()
model_path = args.model_path
model_type = args.model_type


transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # PyTorch's doc tell us to use the mean and standart deviation of the model
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_txt = ['bird', 'boar', 'dog', 'dragon', 'hare', 'horse', 'monkey', 'ox', 'ram', 'rat', 'snake', 'tiger', 'zero']

if model_type == "resnet18":
	model = models.resnet18(pretrained=True)

	# Freeze training for all layers
	for param in model.parameters():
	    param.require_grad = False

	# Newly created modules have require_grad=True by default
	in_feats = model.fc.in_features  # Take the old in_features

	model.fc = nn.Linear(in_feats, len(label_txt))  # Replace the model classifier
	model.fc.require_grad = False

elif model_type == "mobilenet":
	model = models.mobilenet_v2(pretrained=True)

	# Freeze training for all layers
	for param in model.parameters():
	    param.require_grad = False

	# Newly created modules have require_grad=True by default
	in_feats = model.classifier[1].in_features # Take the old in_features

	model.classifier[1] = nn.Linear(in_feats, len(label_txt)) # Replace the model classifier
	model.classifier[1].require_grad = False

else:
	sys.exit("The model type {} is not implemented yet !".format(model_type))

model.load_state_dict(torch.load(model_path, map_location='cpu'))

prediction_txt = "Starting prediction !"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

model.eval()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    transformed_frame = transformations(frame)

    # plt.imshow(transformed_frame.cpu().numpy().transpose(1, 2, 0))
    # plt.show()

    output = model(transformed_frame.unsqueeze(0))
    preds = torch.argmax(output, -1)
    probabilities = F.softmax(output[0], dim=-1)
    prediction_txt = label_txt[preds] + "(" + str(int(probabilities[preds].item() * 100)) + "%)"
    start = time.time()

    cv2.putText(frame, prediction_txt, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('NARUTO', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
