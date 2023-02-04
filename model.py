import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def _visualize_predict(model, x_v, n):
    predictions = np.argmax(model.predict(x_v), axis=-1)
    plt.imshow(x_v[n])
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print('A classe prevista é:', labels[predictions[n]])


mnist = keras.datasets.mnist
(x_t, y_t), (x_v, y_v) = mnist.load_data()
x_t = x_t / 255
x_v = x_v / 255
y_t_cats = to_categorical(y_t)
y_v_cats = to_categorical(y_v)

keras.backend.clear_session()
model = keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_t, y_t_cats, validation_data=(x_v, y_v_cats), epochs=14, batch_size=64)

dados_finais_de_treino = [(l, hist.history[l][-1]) for l in hist.history]
print(dados_finais_de_treino)

plt.figure()
plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.plot(hist.history['accuracy'], label='train_acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.margins(x=0)
plt.margins(y=0)
plt.legend()
plt.show()

for n in range(10):
    _visualize_predict(model, x_v, n)
    plt.show()
