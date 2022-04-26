import atexit
import json
import os
import random
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import  pandas as pd
from tensorboardX import SummaryWriter
import saveCSV
import model
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
saveDir = "d" #directory name
jobs_dir = os.path.join('jobs',saveDir)
snapshot_dir = os.path.join(jobs_dir, 'snapshots')
tensorboard_dir = os.path.join(jobs_dir, 'tensorboardX')
if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
port = 8897

def run_tensorboard(jobs_dir, port=8811):  # for tensorboard
    pid = subprocess.Popen(['tensorboard', '--logdir', jobs_dir, '--host', '0.0.0.0', '--port', str(port)])

    def cleanup():
        pid.kill()

    atexit.register(cleanup)

keypointsPath = 'D:/aiData/keypoints/'
morpList = os.listdir(keypointsPath)
morphemesDict = saveCSV.csvLoad("D:/aiData/morphemesSentenceDict.csv")
morphemes = list(morphemesDict.keys())
morphemesCount = len(morphemesDict.keys())
keypointList = []
for m in morpList:
    p = os.path.join(keypointsPath,m)
    keypoints = os.listdir(p)
    for k in keypoints:
        p = os.path.join(keypointsPath,m,k)
        if os.path.getsize(p) < 8:
            os.remove(p)
    p = os.path.join(keypointsPath, m)
    keypoints = os.listdir(p)
    for k in keypoints:
        kp = os.path.join(keypointsPath,m,k)
        keypointList.append(kp)

####### model initialize #####
epochs = 50
learning_rate = 1e-3
best_loss = -1

run_tensorboard( tensorboard_dir, port)
writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))

fcLayer = model.FClayer().to(device)
model_LSTM = model.LSTM_key().to(device)
feats_all = torch.Tensor().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model_LSTM.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
####### train start ########
for epoch in range(0,epochs):
    random.shuffle(keypointList)
    for k in keypointList:
        kcsv = saveCSV.csvLoad(k)
        m = k.split('\\')[-1].split('-')[0]
        label = morphemesDict[m].iloc[0]
        labelArray = np.zeros(morphemesCount)
        labelArray[label] = 1
        labelArray = torch.from_numpy(labelArray).view(1,318).to(device)
        for i in kcsv.index:
            idx = 0
            karray_r = np.zeros(21 * 2 * 2)
            for r in kcsv.loc[i][:]:
                karray_r[idx] = r.split(',')[0][1:]
                karray_r[idx+1] = r.split(',')[1][1:-1]
                idx = idx+2
            karray_r = torch.from_numpy(karray_r).float().to(device)
            feats = fcLayer(karray_r).to(device)
            feats_all = torch.cat((feats_all, feats), 0)
        output = model_LSTM(feats_all.view(-1, 1, 512))
            #print(list(morphemesSet)[torch.argmax(labels[i])] + "vs" + list(morphemesSet)[torch.argmax(output)])
        #print(labelArray.shape)
        loss = criterion(output,labelArray)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:' + str(epoch) + '_batch:' + str(k.split('\\')[-1].split('-')[0]) + '_loss:' + str(round(loss.item(),3)) + ' <-index: '
              + str(morphemes[output[0].argmax()]))
        feats_all = torch.Tensor().to(device)
        if loss.item()>best_loss:
            best_loss = loss.item()
    writer.add_scalars('train/epoch', {'epoch_loss_high': best_loss}, global_step=epoch)
    best_loss = -1
    scheduler.step()
    # Save model
    torch.save(fcLayer, 'D:/aiData/KeyDetectmodel.pt')
    torch.save(model_LSTM, 'D:/aiData/KeyDetectLSTMmodel.pt')
# Save model
torch.save(fcLayer, 'D:/aiData/KeyDetectmodel.pt')
torch.save(model_LSTM, 'D:/aiData/KeyDetectLSTMmodel.pt')
####### train end #########

###### test start #########
#print(list(morphemesSet)[torch.argmax(output)])