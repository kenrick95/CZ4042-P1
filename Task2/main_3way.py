import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import History
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

## Constants
cal_housing_data = "cal_housing.data"
test_split = 0.3 # split over original data
val_split = 0.5 # split over test data
nfold = 5
epochs = 2500
batch_size = 32
learning_rate = [0.5, 0.3, 0.25, 0.2, 0.1, 0.01, 0.001]
decay = 1e-4
momentum = 0
hidden_nodes = 8

np.random.seed(42)

## Loading data
orig_data = pd.read_csv(cal_housing_data, sep=',')

X_orig = (orig_data.ix[:,0:8].values).astype('float64')
y_orig = (orig_data.ix[:,8].values).astype('float64')

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
    _model.add(Activation('linear'))
    
    _model.add(Dense(1, init='uniform'))
    _model.add(Activation('linear'))

    rmsprop = RMSprop(lr=lr)
    _model.compile(optimizer=rmsprop,
                loss='mse',
                metrics=[])
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
    loss = model.evaluate(X_test, y_test,
        batch_size=batch_size,
        verbose=0)
    
    print("----------- Model lr = %s, hdn = %s " % (lr, hidden_nodes))
    print("loss: %s" % (loss))


    f = open('training_3w_noes_multilr_1hdl_8hdn_pt50_lr%s.pkl' % (lr), 'wb')
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
