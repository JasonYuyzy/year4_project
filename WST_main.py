# -*- encoding: utf-8 -*-
# -------------------------------------------
# Year 4 personal project code work main part
# -------------------------------------------
# Zhengyu Yu

# normal package
import random

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

# arguments setting
args = Options().parse()

# learning rate
args.learning_rateD = 2e-05
args.learning_rateW = 2e-05

# epoch setting
args.epochs = 70

# batch size setting
args.batch_size = 64

# threat items define
args.threat_id = 1

# batch size setting
args.batch_size = 48

args.where_net = "STN"
# batch 48
# args.where_net = "NSTN"

# device define
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if str(device) == "cpu":
    args.GPU_device = False
else:
    args.GPU_device = True
print("Device running:", device)

random.seed(args.random_seeds)
torch.manual_seed(args.random_seeds)
if args.GPU_device:
    torch.cuda.manual_seed_all(args.random_seeds)

# dataset path list loading
fg_path_list, bg_path_list, r_path_list = prepare_data_folders(args)
train_item_path_list = [random.choice(fg_path_list) for i in range(int(len(r_path_list)))]
train_background_path_list = random.sample(bg_path_list, len(r_path_list))
train_real_path_list = r_path_list.copy()

# transformations function
transformations = transforms.Compose([transforms.ToTensor(),
                                      torchvision.transforms.Resize(args.torch_size)])

# dataloader for training
training_set = Dataset(train_item_path_list, train_background_path_list, train_real_path_list, transforms=transformations)
training_generator = data.DataLoader(training_set, shuffle=True, batch_size=args.batch_size, drop_last=True, num_workers=2)

# data loading helper function
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# get the training range
training_range = len(training_generator)
print("training range:", training_range)

# data training iterator helper function
train_iterator = iter(cycle(training_generator))

# load model
if args.where_net == "STN":
    Where_net = Where_stn()
else:
    Where_net = Where_nstn()
Dis_net = DiscriminatorW()

# Apply the weights_init function to randomly initialize all weights
Dis_net.apply(weights_init)

# define the optimizer
# RMSprop optimizer
optimizerW = torch.optim.RMSprop(Where_net.parameters(), lr=args.learning_rateW)
optimizerD = torch.optim.RMSprop(Dis_net.parameters(), lr=args.learning_rateD)

# loss function
# criterion1 = nn.BCELoss()

# continue to train
args.trained = True
start_epoch = 50

# trained model loading
if args.trained:
    checkpoint = torch.load(args.model_dir + '/Wmodel_50_' + args.where_net + '_' + args.threat_item.split('/')[1] + '.h5', map_location=device)
    Dis_net = checkpoint['model_d']
    Where_net = checkpoint['model_g']
    optimizerD.load_state_dict(checkpoint['optimizer_d'])
    optimizerW.load_state_dict(checkpoint['optimizer_g'])

# model to device
Where_net.to(device)
Dis_net.to(device)

# prepare the loss list
G_losses = []
D_losses = []

# start training
print("start training")
for epoch in range(args.epochs):
    print("epoch:", epoch)
    for local_batch in range(training_range):
        fg_imgs, bg_imgs, r_imgs = next(train_iterator)
        # transfer into torch
        fg_imgs, bg_imgs, r_imgs = fg_imgs.float().to(device), bg_imgs.float().to(device), r_imgs.float().to(device)

        # generate affine FG image and the theta value
        stFG_imgs = Where_net(fg_imgs, bg_imgs).detach()
        # FG BG composition
        stFG_masks = torch.where(stFG_imgs < 0.6, 1, 0)
        com_imgs = stFG_imgs * stFG_masks * args.FG_blending + bg_imgs * (1 - stFG_masks * args.BG_blending)

        ####################
        # --- train Discriminator --- #
        ####################
        optimizerD.zero_grad()
        outReals = Dis_net(r_imgs)
        if local_batch % 2 == 0:
            outComps = Dis_net(bg_imgs)
        else:
            outComps = Dis_net(com_imgs)
        loss_D = -torch.mean(Dis_net(r_imgs)) + torch.mean(outComps)
        loss_D.backward()
        optimizerD.step()

        #####################
        # --- train STNs --- #
        #####################
        if epoch % 1 == 0:
            optimizerW.zero_grad()
            # generate affine FG image and the theta value
            stFG_imgs = Where_net(fg_imgs, bg_imgs)
            # FG BG composition
            stFG_masks = torch.where(stFG_imgs < 0.6, 1, 0)
            com_imgs = stFG_imgs * stFG_masks * args.FG_blending + bg_imgs * (1 - stFG_masks * args.BG_blending)

            loss_G = -torch.mean(Dis_net(com_imgs))
            loss_G.backward()
            optimizerW.step()

            # print out the loss
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, args.epochs, local_batch, training_range,
                     loss_D.item(), loss_G.item()))
            if local_batch % 35 == 0 or local_batch % 45 == 0:
                torchvision.utils.save_image(fg_imgs.cpu(), args.sample_dir + args.threat_item + '/WFG_' + str(
                    epoch) + args.where_net + "_" + str(local_batch) + ".png")
                # torchvision.utils.save_image(bg_imgs.cpu(), args.sample_dir + args.threat_item + '/WBG_' + str(epoch) + args.where_net + "_" + str(local_batch) + ".png")
    torchvision.utils.save_image(com_imgs.cpu(), args.sample_dir + args.threat_item + '/WCOM_' + str(
        epoch) + args.where_net + "_" + str(local_batch) + ".png")

    # append the loss for Dis net and STN net
    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    if epoch % 2 == 0:
        state = {'model_d': Dis_net, 'optimizer_d': optimizerD.state_dict(),
                 'model_g': Where_net, 'optimizer_g': optimizerW.state_dict()}
        torch.save(state,
                   args.model_dir + '/Wmodel_' + str(epoch) + '_' + args.where_net + '_' + args.threat_item.split('/')[
                       1] + '.h5')
        print("Model saved in epoch:", epoch)

    x1 = range(len(G_losses))
    y1 = G_losses
    x2 = range(len(D_losses))
    y2 = D_losses

    plt.figure(figsize=(15, 5))
    plt.title("Training loss")
    plt.plot(x1, y1, color="green", label="G(STN) loss")
    plt.plot(x2, y2, color="red", label="D loss")
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.legend()

    plt.savefig(args.sample_dir + '/Wloss_' + args.where_net + ".png")
    plt.close()