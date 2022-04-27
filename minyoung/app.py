
from flask import Flask, render_template, Response
import APIVR
import handdetect_ori
app = Flask(__name__)


@app.route('/')
def hello():
    app.logger.info('index')
    return render_template('plan.html')


@app.route('/service', methods=['GET'])
def service():
    return render_template('service.html')


@app.route('/web')
def func():
    app.logger.info('web')
    return render_template('web.html')


@app.route('/record', methods=['GET'])
def afterrecord():
    return render_template('record.html')


@app.route('/recording', methods=['GET'])
def recording():
    app.logger.info('recording')
    startdata="녹음을 시작합니다!"
    strdata=APIVR.record()
    finishdata="녹음이 끝났습니다!"
    return render_template('recording.html', startdataHtml=startdata, strdataHtml=strdata, finishdataHtml=finishdata)


def startdetect():
    i=0
    while (i < 5):
        handdetect_ori.detect()
        i = i + 1


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(handdetect_ori.detect(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['GET'])
def detect():
    startdetect()
    return render_template('hand_detect.html')


if __name__ == '__main__':
   app.run(host="0.0.0.0")
