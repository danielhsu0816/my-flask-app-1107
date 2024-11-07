import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import cv2 as cv

# trained_model = tf.keras.models.load_model('trained_model.keras')

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


def read_plot_and_test(path_to_img_file):

  # 讀取影像
  test_img_bgr = cv.imread(path_to_img_file)
  test_img_rbg = cv.cvtColor(test_img_bgr, cv.COLOR_BGR2RGB)

  # 視覺化
  plt.figure(figsize=(3, 3))
  plt.imshow(test_img_rbg)
  plt.axis("off")
  plt.show()

  # 將彩色影像轉為灰階影像
  test_img_gray = cv.cvtColor(test_img_rbg, cv.COLOR_RGB2GRAY)

  # 將影像轉換至 48 x 48 的大小，因為是使用 48 x 48 的影像訓練模型
  test_img_resize = cv.resize(test_img_gray, (48, 48), interpolation=cv.INTER_NEAREST)

  # 視覺化
  plt.figure(figsize=(3,3))
  plt.imshow(test_img_resize, cmap="gray")
  plt.axis("off")
  plt.show()

  # reshape 成模型可以接受的輸入形狀
  test_sample = test_img_resize.reshape(1, 48,48,1)

  # 記得 feature scaling !
  test_sample = test_sample / 255.0

  # 交給模型判斷結果
  prediction = trained_model(test_sample).numpy().reshape(-1)
  index = np.argmax(prediction)
  conf = prediction[index]
  print(prediction)
  print("表情為：", Emotion_Rule[index], "，信心", round(conf*100, 2), "%")


if __name__ == "__main__":
    # read_plot_and_test("testImage/test1.jpg")
    help(cv.imread)