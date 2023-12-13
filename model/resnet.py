import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPool2D, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, activations
import numpy as np
import matplotlib.pyplot as plt

# TODO:Data processing
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# class ResNet(Model):
#     def resnet_identity(self, filters):
#         # x_skip = x
#         f1, f2 = filters

#         print("f1", f1, "f2:", f2) #, type(x))

#         out_model = Sequential()

#         out_model.add(Conv2D(f1, kernel_size=(1,1), strides=(1,1), padding="valid"))
#         out_model.add(BatchNormalization())
#         out_model.add(Activation(activations.relu))

#         out_model.add(Conv2D(f1, kernel_size=(3,3), strides=(1,1), padding='same'))
#         out_model.add(BatchNormalization())
#         out_model.add(Activation(activations.relu))

#         out_model.add(Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding="valid"))
#         out_model.add(BatchNormalization())

#         # out_model.add(Add()([x, x_skip])
#         out_model.add(Activation(activations.relu))

#         return out_model

#     def __init__(self) -> None:
#         super(ResNet, self).__init__()
#         self.inside_identity = self.residual_block(filters=[3,1])
#         self.add = Add()
#         self.relu = Activation(activations.relu)

#     def call(self, inputs):
#         x_skip = inputs
#         x = self.inside_identity(inputs)
#         x = self.add([x, x_skip])
#         x = self.relu(x)
#         return x
    
class ResNet2(Model):
    def resnet_identity(self, x, filter):
        x_skip = x
        x = Conv2D(filter, kernel_size=(3,3), padding="same")(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filter, kernel_size=(3,3), padding="same")(x)
        x = BatchNormalization(axis=3)(x)

        x = Add()([x, x_skip])
        x = Activation('relu')(x)

        return x
    def convultion_block(self, x, filter):
        x_skip = x
        x = Conv2D(filter, kernel_size=(3,3), padding="same", strides=(2,2))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filter, kernel_size=(3,3), padding="same")(x)
        x = BatchNormalization(axis=3)(x)

        x_skip = Conv2D(filter, kernel_size=(1,1), strides=(2,2))(x_skip)

        x = Add()([x, x_skip])
        x = Activation('relu')(x)

        return x
    def __init__(self, shape=(32,32,3), classes=10) -> None:
        super(ResNet2, self).__init__()
        x_input = tf.keras.layers.Input(shape)

        x = tf.keras.layers.ZeroPadding2D((3,3))(x_input)
        x = Conv2D(filters=64, kernel_size=7, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding="same")(x)

        block_layers = [3,4,6,3]
        filter_size = 64

        for j in range(block_layers[0]):
            x = self.resnet_identity(x, filter_size)

        for i in range(1,3):
            filter_size *= 2
            x = self.convultion_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = self.resnet_identity(x, filter_size)
        x = AveragePooling2D((2,2), padding="same")(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(classes)(x)

        self.model = Model(inputs = x_input, outputs=x, name="ResNet34")

# model = ResNet2(shape=(28,28,1)).model

# loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name="train_loss")
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

# test_loss = tf.keras.metrics.Mean(name="test_loss")
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")

# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images)
#         loss = loss_obj(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)
#     train_accuracy(labels, predictions)

# @tf.function
# def test_step(images, labels):
#     predictions = model(images)
#     t_loss = loss_obj(labels, predictions)

#     test_loss(t_loss)
#     test_accuracy(labels, predictions)

# EPOCHS = 1

# for epoch in range(EPOCHS):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     test_loss.reset_states()
#     test_accuracy.reset_states()

#     for images, labels in train_ds:
#         train_step(images, labels)
#     for images, labels in test_ds:
#         test_step(images, labels)

#     template = 'Epoch:{}/{} | Loss:{}, Accuracy:{} | Test loss:{}, Test Acc.:{}'
#     print(template.format(
#         epoch+1,
#         EPOCHS,
#         train_loss.result(),
#         train_accuracy.result()*100,
#         test_loss.result(),
#         test_accuracy.result()*100
#     ))
# print(model.summary())