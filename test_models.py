# -*- encoding: utf-8 -*-
# -------------------------------------------
# Year 4 personal project code work test part
# -------------------------------------------
# Zhengyu Yu

# normal package
import cv2
import random
from random import choice

# torch package
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.functional as f

# import functions from other .py files
from options import *
from dataset_load import *
from train_loader import *
from models import *

# load args
args = Options().parse()

###########################
# --- model selection --- #
###########################

# where net selection STN, NSTN and WSTN
args.where_net = "STN"
print("Please input the threat item ID: {1: firearm, 2: firearmparts, 3: knife}")
args.threat_id = int(input())
# threat item selection {1: firearm, 2: firearmparts, 3: knife}
threat_dict = {1: "firearm", 2: "firearmparts", 3: "knife"}
# args.threat_id = 3
args.threat_item = threat_dict[args.threat_id]

###########################
###########################

# load the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device runing:", device)

# load the data path
fg_path_list, bg_path_list, r_path_list = prepare_data_folders(args)

base_path = "./results/trained_model"
# load the model
if True:
    checkpoint = torch.load(args.model_dir + '/model.h5', map_location=device)
    Dis_net = checkpoint['model_d']
    STN_net = checkpoint['model_g']

STN_net.eval()
transformations = transforms.Compose([transforms.ToTensor(),
                                      torchvision.transforms.Resize((512, 512))])
# load the test image
FGimg = Image.open(choice(fg_path_list))
BGimg = Image.open(choice(bg_path_list))

tor_FGimg = transformations(FGimg)
tor_BGimg = transformations(BGimg)
# affine the fg as the training did
tor_FGimg = f.affine(tor_FGimg, angle=0, translate=(0, 0), shear=(0, 0), scale=0.3, fill=1.0)

# change the size to one batch
input_FGimg = tor_FGimg.view(1, 3, 512, 512)
input_BGimg = tor_BGimg.view(1, 3, 512, 512)

# generate affine FG image and detach the output
stFG_imgs = STN_net(input_FGimg, input_BGimg)[0]
# FG BG composite
stFG_masks = torch.where(stFG_imgs < 0.6, 1, 0)
com_imgs = stFG_imgs * stFG_masks * args.FG_blending + tor_BGimg * (1 - stFG_masks*args.BG_blending)

# find the bounding box location
row_list = [pixel[2] for pixel in torch.nonzero(stFG_masks).numpy()]
column_list = [pixel[1] for pixel in torch.nonzero(stFG_masks).numpy()]
x0, y0, x1, y1 = min(row_list), min(column_list), max(row_list), max(column_list)
print("The bounding box location:", (x0, y0), (x1, y1))

test_result = transforms.ToPILImage()(com_imgs).convert("RGB")
big_example = cv2.cvtColor(np.asarray(test_result), cv2.COLOR_RGB2BGR)
# show the binding box
big_example = cv2.rectangle(big_example, (x0, y0), (x1, y1), (0, 255, 0), 2)
cv2.imshow("test_bbox", big_example)
cv2.waitKey(5)

# get the segmentation list
img = np.zeros(args.torch_size, dtype=np.uint8)
# mask_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for pixel in torch.nonzero(stFG_masks).numpy():
    img[pixel[1], pixel[2]] = 200
contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    segmentation = np.array(contour).reshape(int(len(contour)), 2)
    # draw the segmentation
    big_example = cv2.polylines(big_example, np.int32([segmentation]), True, (255, 0, 255), 2)
cv2.imshow("test_seg", big_example)
cv2.waitKey(0)