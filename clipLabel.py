import json
import os
import saveCSV

import cv2
import numpy as np


path_dir = "D:/aiData/수어 영상/1.Training/word/morpheme/02/"
video_path_dir = "D:/aiData/수어 영상/1.Training/word/03/02/"
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
#file_list = resultlist

for f in file_list:
    data = json.load(open(path_dir + f, encoding='UTF8'))
    # morphemes += data['data'][0]['attributes'][0]['name']
    # print(data['data'][0]['attributes'][0]['name'])
    if len(data['data'])<1:
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

    # 재생할 파일의 넓이와 높이
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fileSavePath = os.path.join("D:/aiData/clips_clear", morphemes[fname])
    if not os.path.exists(fileSavePath):
        os.mkdir(fileSavePath)
    fileSavePath = os.path.join(fileSavePath, morphemes[fname]+'-'+fname + ".avi")
    out = cv2.VideoWriter(fileSavePath, fourcc, 30.0, (int(width), int(height)))

    vfps = cap.get(cv2.CAP_PROP_FPS)
    startFrame = int(vfps * morphemesStart[fname])
    endFrame = int(vfps * morphemesEnd[fname])
    frameIndex = 0
    label = np.zeros(morphemesCount)
    label[morphemesDict[morphemes[fname]]] = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    fileCount+=1
    print((fileCount/(morphemesCount*5))*100,"%")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= endFrame:
                break
        else:
              break
cap.release()
out.release()
cv2.destroyAllWindows()