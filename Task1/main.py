import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import History

spambase_data = "spambase.data"

validation_split = 0.3
epochs = 200
batch_size = 32

def zero_mean_normalization(np_matrix):
    """
    Given a numpy matrix, this function will normalize each columns to standard normal distribution
    i.e. for each columns, z = (x - mean) / std_deviation
    """
    means = np.mean(np_matrix, axis=0)
    std_deviations = np.std(np_matrix, axis=0)
    return (np_matrix - means) / std_deviations


train = pd.read_csv(spambase_data, sep=',')

X_train = (train.ix[:,0:56].values).astype('float32')
y_train = (train.ix[:,57].values).astype('float32')

X_train = zero_mean_normalization(X_train)

input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(128, input_dim=input_dim, init='uniform'))
model.add(Activation('sigmoid'))

model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.1)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
    nb_epoch=epochs,
    batch_size=batch_size,
    verbose=2,
    validation_split=validation_split)

loss_and_metrics = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
for i in range(len(model.metrics_names)):
    metric = model.metrics_names[i]
    score = loss_and_metrics[i]
    print("%s: %s" % (metric, score))

