import pandas as pd

dataset = pd.read_excel("./src/pesquisa.xlsx", header=1)
dataset[
    [
        "Tipo",
        "Tipo Endereço",
        "Empreendimento",
        "Zoneamento Novo",
        "Bairro",
        "Cidade",
        "Formato",
        "Topografia",
        "Característica da construção",
    ]
] = dataset[
    [
        "Tipo",
        "Empreendimento",
        "Tipo Endereço",
        "Zoneamento Novo",
        "Bairro",
        "Cidade",
        "Formato",
        "Topografia",
        "Característica da construção",
    ]
].apply(lambda x: x.astype("category").cat.codes)

dataset["CEP"] = dataset["CEP"].apply(lambda x: int(str(x).replace("-", "")))


dataset = dataset.drop(columns=["Melhoramentos Públicos", "RGI", "Estado"])
target = dataset.pop("Valor Venda R$")
dataset = dataset.join(target)
dataset.to_csv("./src/new.csv")
frames = [dataset, dataset, dataset]
dataset = pd.concat(frames)

# regression

from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import os
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Input
import pandas as pd
import io
from math import log
import numpy as np
from keras.regularizers import L1, L2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(dataset.shape)

print(tf.__version__)

dataset = dataset.map(lambda x: np.log(x+0.001))
train, validate, test = np.split(
    dataset.sample(frac=1, random_state=42), [int(0.6 * len(dataset)), int(.8*len(dataset))]
)

pd.DataFrame(test).to_csv('test.csv')

x_train = train.iloc[:, :-1].values
y_train = train["Valor Venda R$"].values
print(y_train)

validation_dataset = validate

x_val = validation_dataset.iloc[:, :-1].values
y_val = validation_dataset["Valor Venda R$"].values
print(y_val)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input((x_train.shape[1],)),
        tf.keras.layers.Dense(50, activation="relu"),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(50, activation="relu"),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(50, activation="relu"),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(50, activation="relu"),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(50, activation="relu"),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="relu"),
    ]
)


num_epochs = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer, loss="MeanSquaredError", metrics=["R2Score"]
)

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=40,
    verbose=1,
    mode="min",
    min_delta=0.001,
    cooldown=0,
    min_lr=0,
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=20,
    steps_per_epoch=10,
    epochs=num_epochs,
    validation_steps=20,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=80, min_delta=0.001, restore_best_weights=True
        ),
        reduce_on_plateau,
    ],
)
model.save("dnn_model.h5")

model.evaluate(x_val, y_val)

acc = history.history["R2Score"]
val_acc = history.history["val_R2Score"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training R2Score")
plt.plot(epochs_range, val_acc, label="Validation R2Score")
plt.legend(loc="lower right")
plt.title("Training and Validation R2Score")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()


