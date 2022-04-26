import atexit
import json
import os
import subprocess
import saveCSV

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from tensorboardX import SummaryWriter

import model
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :",device)
path_dir = "D:/aiData/수어 영상/1.Training/word/morpheme/02/"
video_path_dir = "D:/aiData/수어 영상/1.Training/word/01/"
file_list = os.listdir(path_dir)
file_count = len(file_list)


morphemes = {}
morphemesStart = {}
morphemesEnd = {}
morphemesCount = 0;

resultlist = []
for s in file_list:
    if "F" in s:
        resultlist.append(s)
file_list = resultlist
for f in file_list:
    data = json.load(open(path_dir + f, encoding='UTF8'))
    # morphemes += data['data'][0]['attributes'][0]['name']
    # print(data['data'][0]['attributes'][0]['name'])
    if not data['data']:
        continue
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
saveCSV.csvSave([morphemesDict],"D:/aiData/morphemesDict.csv")
fileCount=0
for fname in morphemes.keys():
    cap = cv2.VideoCapture(video_path_dir + fname)
    if(not cap.isOpened()) :
        print(video_path_dir+fname)
        continue
    vfps = cap.get(cv2.CAP_PROP_FPS)
    startFrame = int(vfps * morphemesStart[fname])
    endFrame = int(vfps * morphemesEnd[fname])
    frameIndex = 0
    label = np.zeros(morphemesCount)
    label[morphemesDict[morphemes[fname]]] = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    fc = 0
    fileCount+=1
    print((fileCount/morphemesCount)*100,"%")
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
            #cv2.imshow('img', img_r)
            saveName = str("D:/aiData/clips/"
                               +str(morphemesDict[morphemes[fname]])
                               +"-"
                               +str(endFrame-startFrame)
                               +"-"
                               +str(fc)
                               +".jpg")
            #print(saveName)
            r = cv2.imwrite(saveName, img_r)
            #print(r)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= endFrame:
                break
        else:
              break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        fc+=1
