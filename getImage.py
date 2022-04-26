import atexit
import json
import os
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from tensorboardX import SummaryWriter

import model
import torch
import torch.nn as nn

import saveCSV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_dir = "D:/aiData/수어 영상/1.Training/word/morpheme/01/"
video_path_dir = "D:/aiData/clips/"
video_list = os.listdir(video_path_dir)

morphemes = saveCSV.csvLoad("D:/aiData/morphemesDict.csv")
morphemes = list(morphemes.keys())
print(morphemes)

for f in video_list:
    img = cv2.imread(video_path_dir+f)
    info = f.split('.')[0].split('-')
    label = info[0]
    clip_len = info[1]
    clip_count = info[2]
    #print(label,clip_len,clip_count)