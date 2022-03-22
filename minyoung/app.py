
from flask import Flask, render_template
import APIVR
app = Flask(__name__)


@app.route('/')
def hello():
    app.logger.info('index')
    return render_template('web.html')


@app.route('/wave', methods=['GET'])
def recording():
    app.logger.info('/wave')
    APIVR.record()
    return render_template('wave.html')

#
# @app.route('/detect', methods=['GET'])
# def recording():
#     return render_template('hand_detect.html')


if __name__ == '__main__':
   app.run(host="0.0.0.0")
