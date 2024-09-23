import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

## 输入维度
input_dim = 784  # 28x28 pixels for MNIST编码器
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation="relu")(input_layer)
latent = Dense(32, activation="relu")(encoder)

## 解码器
decoder = Dense(128, activation="relu")(latent)
output_layer = Dense(input_dim, activation="sigmoid")(decoder)

## 自动编码器模型
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adam", loss="mse")

## 加载MNIST数据集
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

## 训练noise的随机初始化
noise_train = tf.keras.backend.random_normal(shape=tf.shape(x_train))
noise_test = tf.keras.backend.random_normal(shape=tf.shape(x_test))

## 训练模型
# autoencoder.fit(
#     x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test)
# )

autoencoder.fit(
    noise_train,
    x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(noise_test, x_test),
)

## 保存模型
save_dir = "/home/weihaizhi/Desktop/generated_model"
os.makedirs(save_dir, exist_ok=True)
autoencoder.save(f"{save_dir}/ae.keras")

## 测试模型
model = keras.models.load_model(f"{save_dir}/ae.keras")
predict_output = model.predict(noise_test)
print(predict_output)


## 可视化 x_test 和 predict_output
def plot_comparison(x_test, predict_output, num_examples=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_examples):
        # 展示原始图片
        ax = plt.subplot(2, num_examples, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # 展示预测输出图片
        ax = plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.imshow(predict_output[i].reshape(28, 28), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()


# 调用可视化函数
plot_comparison(x_test, predict_output, num_examples=10)
