import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 加载训练好的模型
model = tf.keras.models.load_model("mnist_model.keras")


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 读取手写数字图片
img_path = "1.png"
img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
# img_show(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x.astype('float32') / 255

# 进行预测
prediction = model.predict(x)
predicted_label = np.argmax(prediction[0])

print(f"预测的数字标签为：{predicted_label}")
