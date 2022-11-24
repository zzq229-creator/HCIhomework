from ffmpy import FFmpeg as mpy
import os
import app
from PIL import Image


def trans_to_wav(mp3_path):
    '''
    格式转换格式
    :param mp3_file:
    :param wav_folder:
    :return:
    '''
    wav_path = 'tmp.wav'
    # 格式化文件
    cmder = '-f wav -ac 1 -ar 16000'
    # 创建转换器对象
    mpy_obj = mpy(
        executable='ffmpeg.exe',
        inputs={
            mp3_path: None
        },
        outputs={
            wav_path: cmder
        }
    )
    mpy_obj.run()


if __name__ == '__main__':
    import numpy as np
    import cv2
    img = Image.open("face/zzq.jpg")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = app.detectionmaskface(img)
    print(res)
    pass
