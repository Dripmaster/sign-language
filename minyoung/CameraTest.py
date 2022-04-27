import model
import torch
#import saveCSV
import cv2
import mediapipe as mp
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


model_key = torch.load('/Users/yeongmin/Downloads/signProject/sign-language-main/KeyDetectmodel.pt',map_location=torch.device('cpu'))
model_key.eval()
model_lstm = torch.load('/Users/yeongmin/Downloads/signProject/sign-language-main/KeyDetectLSTMmodel.pt',map_location=torch.device('cpu'))
model_lstm.eval()

predic_result = np.array([0,1])

def predictProcess(k):
    feats_all = torch.Tensor()
    for i in range( len(k)):
        idx = 0
        karray_r = np.zeros(21 * 2 * 2)
        idx = 0
        for r in range(42):
            karray_r[idx] = np.array(k[i][r][0])
            karray_r[idx+1] = np.array(k[i][r][1])
            idx+=2
        karray_r = torch.from_numpy(karray_r).float()
        feats = model_key(karray_r)
        feats_all = torch.cat((feats_all, feats), 0)
    output = model_lstm(feats_all.view(-1, 1, 512))
    feats_all = torch.Tensor()
    return output

failFlag = 0
cap = cv2.VideoCapture(0)

vfps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))  # 가로 길이 가져오기
height = int(cap.get(4))  # 세로 길이 가져오기
handsType = []
myHands = []
errorFrame = []
myHand = []
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        failFlag = 0
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            hc = len(handsType)
            tempHandsType = list()
            errorHand = ""
            for hand_landmarks in results.multi_hand_landmarks:
                myHand = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    myHand.append((int(lm.x * width), int(lm.y * height)))
                myHands.append(myHand)

            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                handsType.append(handType)
                tempHandsType.append(handType)
            if len(tempHandsType) > 1 and tempHandsType[0] == tempHandsType[1]:
                handsType[-1] = 'Right'
                handsType[-2] = 'Left'
                tempHandsType[0] = 'Left'
                tempHandsType[1] = 'Right'
            if len(handsType) - hc < 2:
                if tempHandsType[0] == 'Left':  # 오른손 없음
                    errorHand = 'Right'

                    lastHand = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                            (0, 0)]
                    myHands.append(lastHand)
                    handsType.append('Right')
                else:  # 왼손 없음
                    errorHand = 'Left'

                    lastHand = myHands[-1].copy()
                    myHands[-1] = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                               (0, 0), (0, 0), (0, 0)]
                    myHands.append(lastHand)
                    handsType[-1] = 'Left'
                    handsType.append('Right')
                errorFrameCount = len(handsType) / 2 - 1
                # print("양손검출실패:",errorHand,"/ frame:",errorFrameCount)
                errorFrame.append((errorFrameCount, errorHand))
            elif not (tempHandsType[0] == 'Left' and tempHandsType[1] == 'Right'):
                # print("왼손오른손순서 틀림:",tempHandsType,"/ frame:",len(handsType)/2-1) # Right-Left
                tempHand = myHands[-1].copy()  # Left
                myHands[-1] = myHands[-2].copy()
                myHands[-2] = tempHand
                handsType[-1] = 'Right'
                handsType[-2] = 'Left'
        else:
            #print("detect fail:")
            failFlag = 1
        #if failFlag == 1:
        #    continue

        for f, d in errorFrame:
            fc = int(f * 2)
            if d == 'Right':
                fc = int(f * 2) + 1
            for i in range(len(myHands[fc])):
                c = len(myHands)
                if c <= fc + 2:
                    myHands[fc][i] = (int((myHands[fc - 2][i][0])),
                                      int((myHands[fc - 2][i][1])))
                elif 0 > fc - 2:
                    myHands[fc][i] = (int((myHands[fc + 2][i][0])),
                                      int((myHands[fc + 2][i][1])))
                else:
                    myHands[fc][i] = (int((myHands[fc - 2][i][0] + myHands[fc + 2][i][0]) / 2),
                                      int((myHands[fc - 2][i][1] + myHands[fc + 2][i][1]) / 2))
        normHands = []
        notnormHands = myHands
        for f in myHands:
            maxDisX = 1
            maxDisY = 1
            normHand = []
            for i in range(len(f)):

                disX = f[i][0] - f[0][0]
                if abs(maxDisX) < abs(disX):
                    maxDisX = abs(disX)

                disY = f[i][1] - f[0][1]
                if abs(maxDisY) < abs(disY):
                    maxDisY = abs(disY)

            for k in range(len(f)):
                #f[k] = (((f[k][0] - f[0][0]) / maxDisX), ((f[k][1] - f[0][1]) / maxDisY))
                disX = (f[k][0] - f[0][0])/maxDisX
                disY = (f[k][1] - f[0][1])/maxDisY
                normHand.append((disX,disY))
                #f[k] = (disX,disY)
            normHands.append(normHand)
        if (len(myHands) > 2):
            for h in myHands[-2]:
                if failFlag == 0:
                    print(h)
                    cv2.circle(image, (h[0], h[1]), 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.imshow("title", image)
        myHands = normHands
        handkeypoints = dict()
        idx = 0
        for i in range(int(len(myHands) / 2)):
            handkeypoints[idx] = myHands[i * 2] + myHands[i * 2 + 1]
            idx += 1
        if(len(handkeypoints)>30*3):
            predic_result = predictProcess(handkeypoints)
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

print("result:",  predic_result.argmax().item())

cap.release()
#out.release()
cv2.destroyAllWindows()