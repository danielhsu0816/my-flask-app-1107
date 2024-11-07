import tensorflow as tf

# 載入 .keras 模型
model = tf.keras.models.load_model("trained_model.keras")

# 將模型轉換為 tflite 格式
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 將轉換後的模型儲存成 .tflite 文件
with open("trained_model.tflite", "wb") as f:
    f.write(tflite_model)
