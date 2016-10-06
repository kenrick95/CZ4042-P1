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
# learning_rate = [0.5, 0.25, 0.1, 0.01, 0.001, 1e-4]
learning_rate = [0.5, 0.3, 0.25, 0.2, 0.1, 0.01, 0.001]
# learning_rate = [0.1]#[0.5, 0.3, 0.25, 0.2, 0.1, 0.01, 0.001]
hdl = [1]#[0, 1, 2, 3, 4]
decay = 1e-4
momentum = 0
hidden_nodes = [8]#[4, 8, 12, 16, 20, 24]

np.random.seed(42)

## Loading data
orig_data = pd.read_csv(cal_housing_data, sep=',')

X_orig = (orig_data.ix[:,0:8].values).astype('float64')
y_orig = (orig_data.ix[:,8].values).astype('float64')

## Normalization
def train_test_normalization(train_matrix, test_matrix):
    means = np.mean(train_matrix, axis=0)
    std_deviations = np.std(train_matrix, axis=0)
    return (train_matrix - means) / std_deviations, (test_matrix - means) / std_deviations

## Data split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_orig, y_orig, test_size=test_split)
X_train, X_test = train_test_normalization(X_train, X_test)

## Build model
input_dim = X_train.shape[1]

def create_model(lr, hdl, hd_nodes):
    _model = Sequential()

    hdn = hd_nodes
    if (hdl == 0):
        hdn = 1
    _model.add(Dense(hdn, input_dim=input_dim, init='uniform'))
    _model.add(Activation('linear'))

    for i in range(1, hdl):
        _model.add(Dense(hdn, init='uniform'))
        _model.add(Activation('linear'))
    
    if hdl > 0:
        _model.add(Dense(1, init='uniform'))
        _model.add(Activation('linear'))

    rmsprop = RMSprop(lr=lr)
    _model.compile(optimizer=rmsprop,
                loss='mse',
                metrics=[])
    return _model

best_model = None
best_model_loss = float("inf")
best_model_index = -1
index = 0

for lr in learning_rate:
    for _hdl in hdl:
        for _hdn in hidden_nodes:
            model = None
            avg_loss = 0
            total_loss = 0
            ## K-fold cross validation
            kf = KFold(len(y_train), n_folds=nfold, shuffle=True)
            for train, validation in kf:
                model = create_model(lr, _hdl, _hdn)

                ## Train model
                history = model.fit(X_train[train], y_train[train],
                    nb_epoch=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    validation_data=(X_train[validation], y_train[validation]),
                    verbose=2)

                ## Validate model
                loss = model.evaluate(X_train[validation], y_train[validation],
                    batch_size=batch_size,
                    verbose=0)
                print("----------- Model index %s; lr = %s, hdn = %s, hdl = %s; val_loss = %s " % (index, lr, _hdn, _hdl, loss))
                total_loss += loss
            avg_loss = total_loss / nfold
            print("Model index %s; lr = %s, hdn = %s, hdl = %s; avg val_loss = %s " % (index, lr, _hdn, _hdl, avg_loss))

            if avg_loss < best_model_loss or index == 0:
                best_model_loss = avg_loss
                best_model = pickle.dumps(model)
                best_model_index = index
            index += 1


print("-----------------  Best model (index: %s) re-training...." % best_model_index)
best_model = pickle.loads(best_model)
history = best_model.fit(X_train, y_train,
        nb_epoch=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        validation_data=(X_test, y_test),
        verbose=2)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print("----------- Model %s: loss_and_metrics: %s " % (index, loss_and_metrics))
# f = open('training_3w_multihdn_noEs_%shdl_%shdn_lr%s.pkl' % (_hdl, _hdn, lr), 'wb')
# pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()
