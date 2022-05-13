from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
# Додамо шар Embedding чекаючи на вході словник розміру 1000, і
# На виході вкладення розмірністю 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Додамо шар LSTM з 128 внутрішніми вузлами.
model.add(layers.LSTM(128))

# Додамо шар Dense з 10 вузлами і активацією softmax.
model.add(layers.Dense(10))

model.summary()


model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Виходом GRU буде 3D тензор розміру (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# Виходом SimpleRNN буде 2D тензор розміру (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))

model.summary()


encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None, ))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

# Повертає стан в додаток до вихідних даних
output, state_h, state_c = layers.LSTM(
    64, return_state=True, name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None, ))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

# Передає 2 стани в новий шар LSTM в якості початкового стану
decoder_output = layers.LSTM(
    64, name='decoder')(decoder_embedded, initial_state=encoder_state)
output = layers.Dense(10)(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output)
model.summary()

paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() скине кешований стан до початкового initial_state.
# Якщо initial_state не було задано, по замовчуванню будуть використані нульові стани.
lstm_layer.reset_states()


model = tf.keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True),
input_shape=(5, 10)))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10))

model.summary()


batch_size = 64
# Кожен пакет зображення MNIST це тензор розмірністю (batch_size, 28, 28).
# Кожна вхідна послідовність розміру (28, 28) (висота розглядається як час).
input_dim = 28
units = 64
output_size = 10 # метки от 0 до 9
# Побудуємо RNN модель
def build_model(allow_cudnn_kernel=True):
    # CuDNN доступний тільки на рівні шару, а не на рівні осередку.
    # Це означає `LSTM (units)` буде використовувати ядро CuDNN,
    # тоді як RNN (LSTMCell (units)) буде використовувати non-CuDNN ядро.
    if allow_cudnn_kernel:
# Шар LSTM з параметрами за замовчуванням використовує CuDNN.

        lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
# Обгортка LSTMCell шаром RNN не використовуватиме CuDNN.
        lstm_layer = tf.keras.layers.RNN( tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))
    model = tf.keras.models.Sequential([ lstm_layer,
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(output_size)])
    return model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

model = build_model(allow_cudnn_kernel=True)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)


slow_model = build_model(allow_cudnn_kernel=False)
slow_model.set_weights(model.get_weights())
slow_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   optimizer='sgd',
                   metrics=['accuracy'])
slow_model.fit(x_train, y_train,
               validation_data=(x_test, y_test),
               batch_size=batch_size,
               epochs=1)

with tf.device('CPU:0'):
  cpu_model = build_model(allow_cudnn_kernel=True)
  cpu_model.set_weights(model.get_weights())
  result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
  print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
  plt.imshow(sample, cmap=plt.get_cmap('gray'))

  NestedInput = collections.namedtuple('NestedInput', ['feature1', 'feature2'])
  NestedState = collections.namedtuple('NestedState', ['state1', 'state2'])


class NestedCell(tf.keras.layers.Layer):

    def __init__(self, unit_1, unit_2, unit_3, **kwargs):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.unit_3 = unit_3
        self.state_size = NestedState(state1=unit_1,
                                    state2=tf.TensorShape([unit_2, unit_3]))
        self.output_size = (unit_1, tf.TensorShape([unit_2, unit_3]))
        super(NestedCell, self).__init__(**kwargs)


    def build(self, input_shapes):
        input_1 = input_shapes.feature1[1]
        input_2, input_3 = input_shapes.feature2[1:]

        self.kernel_1 = self.add_weight(
            shape=(input_1, self.unit_1), initializer='uniform', name='kernel_1')
        self.kernel_2_3 = self.add_weight(
            shape=(input_2, input_3, self.unit_2, self.unit_3),
            initializer='uniform',
            name='kernel_2_3')

    def call(self, inputs, states):
        input_1, input_2 = tf.nest.flatten(inputs)
        s1, s2 = states

        output_1 = tf.matmul(input_1, self.kernel_1)
        output_2_3 = tf.einsum('bij,ijkl->bkl', input_2, self.kernel_2_3)
        state_1 = s1 + output_1
        state_2_3 = s2 + output_2_3

        output = [output_1, output_2_3]
        new_states = NestedState(state1=state_1, state2=state_2_3)

        return output, new_states


unit_1 = 10
unit_2 = 20
unit_3 = 30

input_1 = 32
input_2 = 64
input_3 = 32
batch_size = 64
num_batch = 100
timestep = 50

cell = NestedCell(unit_1, unit_2, unit_3)
rnn = tf.keras.layers.RNN(cell)

inp_1 = tf.keras.Input((None, input_1))
inp_2 = tf.keras.Input((None, input_2, input_3))

outputs = rnn(NestedInput(feature1=inp_1, feature2=inp_2))

model = tf.keras.models.Model([inp_1, inp_2], outputs)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
unit_1 = 10
unit_2 = 20
unit_3 = 30

input_1 = 32
input_2 = 64
input_3 = 32
batch_size = 64
num_batch = 100
timestep = 50

cell = NestedCell(unit_1, unit_2, unit_3)
rnn = tf.keras.layers.RNN(cell)

inp_1 = tf.keras.Input((None, input_1))
inp_2 = tf.keras.Input((None, input_2, input_3))

outputs = rnn(NestedInput(feature1=inp_1, feature2=inp_2))

model = tf.keras.models.Model([inp_1, inp_2], outputs)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


input_1_data = np.random.random((batch_size * num_batch, timestep, input_1))
input_2_data = np.random.random((batch_size * num_batch, timestep, input_2, input_3))
target_1_data = np.random.random((batch_size * num_batch, unit_1))
target_2_data = np.random.random((batch_size * num_batch, unit_2, unit_3))
input_data = [input_1_data, input_2_data]
target_data = [target_1_data, target_2_data]

model.fit(input_data, target_data, batch_size=batch_size)