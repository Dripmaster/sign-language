import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import model
import torch
import torch.nn as nn

path_dir = "C:/Users/son34/Documents/공문/004.수어_영상_sample/004.수어_영상_sample/라벨링데이터/morpheme/"
video_path_dir = "C:/Users/son34/Documents/공문/004.수어_영상_sample/004.수어_영상_sample/원시데이터/"
file_list = os.listdir(path_dir)
file_count = len(file_list)

morphemes = {}
morphemesStart = {}
morphemesEnd = {}
morphemesCount = 0;
for f in file_list:
    data = json.load(open(path_dir + f, encoding='UTF8'))
    # morphemes += data['data'][0]['attributes'][0]['name']
    # print(data['data'][0]['attributes'][0]['name'])
    morphemes[data['metaData']['name']] = data['data'][0]['attributes'][0]['name']
    morphemesStart[data['metaData']['name']] = data['data'][0]['start']
    morphemesEnd[data['metaData']['name']] = data['data'][0]['end']

morphemesSet = set(morphemes.values())
morphemesDict = dict()
i = 0
for m in morphemesSet:
    morphemesDict[m] = i
    i = i+1
morphemesCount = len(morphemesSet)

#for m in morphemesSet:
#    print(m)

#print(morphemesCount)
# print(data['metaData']['name'])
# print(len(data['data']))

# print(data['data'][0]['attributes'][0]['name'])
# print(data['data'][0]['start'])
# print(data['data'][0]['end'])
videos = list()
labels = list()
flagList = list()
fileCount=0
for fname in morphemes.keys():
    cap = cv2.VideoCapture(video_path_dir + fname)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    startFrame = vfps * morphemesStart[fname]
    endFrame = vfps * morphemesEnd[fname]
    frameIndex = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame + frameIndex*8)
        m_video = list()
        rvideo = list()
        gvideo = list()
        bvideo = list()
        fc = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                b, g, r, a = 255, 255, 255, 0
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil = Image.fromarray(cv2.resize(frame,(512,512)))
                draw = ImageDraw.Draw(img_pil)
                draw.text((60, 70), morphemes[fname], font=font, fill=(b, g, r, a))
                img = np.array(img_pil)
                img_r = cv2.resize(frame,(128,128))
                #cv2.imshow('img_r', img_r)
                cv2.imshow('img', img)
                b,g,r = cv2.split(img_r)
                b = torch.tensor(b)
                g = torch.tensor(g)
                r = torch.tensor(r)
                imgTensor = torch.stack((b,g,r),dim=0)
                m_video.append(imgTensor)
                rvideo.append(r)
                gvideo.append(g)
                bvideo.append(b)
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= endFrame:
                    break
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fc+=1
            if fc>15 :
                break
        rvideo = torch.stack(rvideo)/255
        gvideo = torch.stack(gvideo)/255
        bvideo = torch.stack(bvideo)/255
        m_video = torch.stack((rvideo,gvideo,bvideo),dim = 0)
        videos.append(m_video)
        label = torch.zeros(morphemesCount)
        label[morphemesDict[morphemes[fname]]] = 1
        labels.append(label)
        frameIndex +=1
        if endFrame-(startFrame + frameIndex*8) < 16:
            flagList.append(1)
            break
        else:
            flagList.append(0)
    fileCount+=1
    if fileCount>19:
        break
videos = torch.stack(videos) #clip
labels = torch.stack(labels) #output
####### model initialize #####
c3d = model.C3D()
model_LSTM = model.LSTM_anno()
feats_all = torch.Tensor()
####### train start ########
for i in range(0,videos.size(dim=0)):
    feats = c3d(videos[i].view(1,3,16,128,128))
    feats_all = torch.cat((feats_all, feats), 0)
    if flagList[i] == 1:
        output = model_LSTM(feats_all.view(-1,1,4096))
        feats_all = torch.Tensor()
        print( list(morphemesSet)[torch.argmax(labels[i])]+"vs"+list(morphemesSet)[torch.argmax(output)])
        #loss = criterion(scores,labeled)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

####### train end #########

###### test start #########
#print(list(morphemesSet)[torch.argmax(output)])