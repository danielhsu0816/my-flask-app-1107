import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)
# app = Flask(__name__, template_folder='my_templates', static_folder='my_static')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # 限制最大上傳檔案大小為 1MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 載入訓練好的模型
trained_model = tf.keras.models.load_model('trained_model.keras')

# 為此 Dataset 定義的 mapping 關係
Emotion_Rule = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def read_plot_and_test(image_path):
    """
    opencv函數預設會以BGR (Blue-Green-Red) 的格式返回影像數據，而不是常見的 RGB (Red-Green-Blue) 格式。
    一張彩色圖片預設的顏色通道有三個，差異只在排序順序的不同，而影像處理的習慣是以RGB來做處理
    R: 0~255
    G: 0~255
    B: 0~255
    每個顏色的色階由1~256表示，存在陣列中會是0~255對應之。

    help(cv2) 可以查看 CV2的模組說明
    或是可以到官方網站查看 function 說明文件
    https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
    """
    
    # 讀取影像
    test_img_bgr = cv.imread(image_path) 
    test_img_rbg = cv.cvtColor(test_img_bgr, cv.COLOR_BGR2RGB)

    # # 視覺化
    # plt.figure(figsize=(3, 3))
    # plt.imshow(test_img_rbg)
    # plt.axis("off")
    # plt.show()

    # 將彩色影像轉為灰階影像
    test_img_gray = cv.cvtColor(test_img_rbg, cv.COLOR_RGB2GRAY)

    # 將影像轉換至 48 x 48 的大小 # 可參考: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
    test_img_resize = cv.resize(test_img_gray, (48, 48), interpolation=cv.INTER_NEAREST)

    # reshape 成模型可以接受的輸入形狀 (batch_size, height, width, channels) batch_size=>一個批次有幾張圖 / 1 channels 表示灰階影像（單通道）
    test_sample = test_img_resize.reshape(1, 48, 48, 1)
    
    # 記得 feature scaling  也就是說縮放數值區間到0~1之間 
    test_sample = test_sample / 255.0

    # 交給模型判斷結果 
    # reshape(-1) 用於將輸出攤平成一維陣列，方便後續的處理。
    # 類似把 2x2 多為矩陣 轉乘 (4,) 一維矩陣
    prediction = trained_model.predict(test_sample).reshape(-1)
    index = np.argmax(prediction) # Returns the indices of the maximum values along an axis. 
    # 如果模型預測有三個類別（0, 1, 2），且 prediction 為 [0.1, 0.7, 0.2]，則 argmax 會返回 1
    conf = prediction[index] # 如上所示，返回 最可能的結果的預測機率 信心值
    
    #  decimal digits 第二個參數為保留小數點後幾位
    return Emotion_Rule[index], round(conf * 100, 2) 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Http Method 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 上傳檔案
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # 儲存檔案
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 使用模型進行預測
            emotion, confidence = read_plot_and_test(file_path)

            return render_template('result.html', emotion=emotion, confidence=confidence, filename=filename)

    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST': 
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file: 
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return '檔案上傳成功！'

    return render_template('upload.html')   

if __name__ == "__main__":
    # 確保上傳資料夾存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    os.makedirs( 'static/testfolder/', exist_ok=True)
    app.run(debug=True)
    # app.run(debug=True , port=8000)
