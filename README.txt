There two training file for training the ST-GANs:
    1 training_main.py: for training the data-binding STNs (with args.where_net = STN) and feature-binding STNs (with args.where_net = NSTN), both of them training with the traditional GANs optimisation.
    2 WST_main.py: for training the data-binding STNs with the WGANs' optimisation mode.

The dataset storing in the file called: threat_dataset, which including the real image set, background image set and foreground image set
Download through the link: https://drive.google.com/drive/folders/1NWFmZf_cxr9KMqWL_KE-M88JC9ybhIXD?usp=sharing

models.py: all the STNs and Dis net
dataset_load.py: reading and returning the data path list.
options.py: all the default settings inside
train_loader.py: the data loader for the training

The model information will be saved in the file path: ./result/trained_model/
The training result/samples will be saved in the file path: ./result/samples/firearm(or firearmparts or knife)
The test model is not the optimise one, just for simply testing whether the model, the bounding box and the segmentation could work.

test_models.py: can change the dataset name and run it directly to see the results

The package requirement for python:
    argparse, random, cv2, json, PIL, matplotlib, torch, torchvision