import pyaudio
import wave
import requests
import json
import numpy as np
import konlpy
from konlpy.tag import Hannanum, Kkma, Komoran
from decouple import config

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"\



def test():
    print("test")


def record():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    start="녹음을 시작합니다!"
    print(start)
    # print("Start to record the audio")ㅋ


    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    finish="녹음이 끝났습니다!"
    print(finish)
    # print("Recording is finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    kakao_speech_url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"

    key = config('key')

    headers = {
        "Content-Type": "application/octet-stream",
        "X-DSS-Service": "DICTATION",
        "Authorization": "KakaoAK " + key,
    }

    with open('./file.wav', 'rb') as fp:
        audio = fp.read()

    res = requests.post(kakao_speech_url, headers=headers, data=audio)
    # print("res.text : " + res.text)

    result_json_string = res.text[res.text.index('{"type":"finalResult"'):res.text.rindex('}') + 1]
    result = json.loads(result_json_string)
    # print("result")
    # print(result)
    print('음성 녹음 결과')
    print(result['value'])
    morpheme(result['value'])
    result2='"'+ result['value']+'"'
    # return result['value']
    return result2


# 텍스트 정제 형태소 분리
def morpheme(result):
    hannaum = Hannanum()
    print(hannaum.morphs(result))

