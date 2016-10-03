import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.cross_validation import KFold, StratifiedKFold
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import History
import pickle
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)

## Constants
spambase_data = "spambase.data"
test_split = 0.3 # split over original data
val_split = 0.5
epochs = 2500
batch_size = 32
learning_rate = [0.5, 0.25, 0.2, 0.1, 0.01, 0.001, 1e-4]
decay = 1e-4
momentum = 0
hidden_nodes = 20

np.random.seed(42)

## Loading data
orig_data = pd.read_csv(spambase_data, sep=',')

X_orig = (orig_data.ix[:,0:56].values).astype('float32')
y_orig = (orig_data.ix[:,57].values).astype('float32')

## Normalization
def train_test_val_normalization(train_matrix, test_matrix, val_matrix):
    means = np.mean(train_matrix, axis=0)
    std_deviations = np.std(train_matrix, axis=0)
    return (train_matrix - means) / std_deviations, (test_matrix - means) / std_deviations, (val_matrix - means) / std_deviations

## Data split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_orig, y_orig, test_size=test_split)
X_test, X_val, y_test, y_val = cross_validation.train_test_split(X_test, y_test, test_size=val_split)
X_train, X_test, X_val = train_test_val_normalization(X_train, X_test, X_val)

## Build model
input_dim = X_train.shape[1]

def create_model(lr):
    _model = Sequential()
    _model.add(Dense(hidden_nodes, input_dim=input_dim, init='uniform'))
    _model.add(Activation('sigmoid'))

    _model.add(Dense(hidden_nodes, init='uniform'))
    _model.add(Activation('sigmoid'))
    
    _model.add(Dense(hidden_nodes, init='uniform'))
    _model.add(Activation('sigmoid'))

    _model.add(Dense(1, init='uniform'))
    _model.add(Activation('sigmoid'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum)
    _model.compile(optimizer=sgd,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return _model

for lr in learning_rate:
    model = None
    model = create_model(lr)

    ## Train model
    history = model.fit(X_train, y_train,
        nb_epoch=epochs,
        batch_size=batch_size,
        # callbacks=[early_stopping],
        validation_data=(X_val, y_val),
        verbose=2)

    ## Validate model
    loss_and_metrics = model.evaluate(X_test, y_test,
        batch_size=batch_size,
        verbose=0)
    
    print("----------- Model lr = %s " % lr)
    for i in range(len(model.metrics_names)):
        metric = model.metrics_names[i]
        score = loss_and_metrics[i]
        print("%s: %s" % (metric, score))


    f = open('training_model_3way_2500ep_3hdl_lr%s.pkl' % (lr), 'wb')
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
