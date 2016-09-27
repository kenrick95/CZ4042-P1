import numpy as np
import pandas as pd
import pickle
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import History
from keras.wrappers.scikit_learn import KerasRegressor

## Constants
cal_housing_data = "cal_housing.data"
test_split = 0.3 # split over original data
nfold = 5
epochs = 2500
batch_size = 32
learning_rate = 0.1
decay = 1e-4
momentum = 0
hidden_nodes = 4

np.random.seed(42)

## Loading data
orig_data = pd.read_csv(cal_housing_data, sep=',')

X_orig = (orig_data.ix[:,0:7].values).astype('float64')
y_orig = (orig_data.ix[:,8].values).astype('float64')

## Normalization
def zero_mean_normalization(np_matrix):
    """
    Given a numpy matrix, this function will normalize each columns to standard normal distribution
    i.e. for each columns, z = (x - mean) / std_deviation
    """
    means = np.mean(np_matrix, axis=0)
    std_deviations = np.std(np_matrix, axis=0)
    return (np_matrix - means) / std_deviations

X_orig = zero_mean_normalization(X_orig)
# y_orig = zero_mean_normalization(y_orig)

## Data split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_orig, y_orig, test_size=test_split)

print("y_train: mean(%.2f), std_dev(%.2f)" % (np.mean(y_train), np.std(y_train)))

## Build model
input_dim = X_train.shape[1]

def create_model():
    _model = Sequential()

    _model.add(Dense(hidden_nodes, input_dim=input_dim, init='uniform'))
    _model.add(Activation('linear'))
    
    _model.add(Dense(1, init='uniform'))
    _model.add(Activation('linear'))

    # sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum)
    _model.compile(optimizer='adam',
                loss='mse',
                metrics=[])
    return _model

best_model = None
best_model_loss = float("inf")
best_model_index = -1
index = 0

# estimator = KerasRegressor(build_fn=create_model, nb_epoch=epochs, batch_size=batch_size, verbose=0)
# kf = KFold(len(y_train), n_folds=nfold, shuffle=True)
# results = cross_val_score(estimator, X_train, y_train, cv=kf)
# print(results)
# print("Results: %.5f (%.5f) MSE" % (results.mean(), results.std()))

# exit()

## K-fold cross validation
kf = KFold(len(y_train), n_folds=nfold, shuffle=True)
for train, validation in kf:
    model = create_model()

    ## Train model
    history = model.fit(X_train[train], y_train[train],
        nb_epoch=epochs,
        batch_size=batch_size,
        validation_data=(X_train[validation], y_train[validation]),
        verbose=0)

    ## Validate model
    loss_and_metrics = model.evaluate(X_train[validation], y_train[validation],
        batch_size=batch_size,
        verbose=2)
    
    print("----------- Model %s: loss_and_metrics: %s " % (index, loss_and_metrics))
    
    ## Select best model
    if loss_and_metrics < best_model_loss or index == 0:
        best_model_loss = loss_and_metrics
        best_model = model
        best_model_index = index
    index += 1

print("-----------------  Best model (index: %s) re-training...." % best_model_index)
## Train best model on whole train set
history = best_model.fit(X_train, y_train,
        nb_epoch=epochs,
        batch_size=batch_size,
        verbose=0)

## Evaluate model
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print("----------- Model %s: loss_and_metrics: %s " % (index, loss_and_metrics))

f = open('model_best_2500ep_4hdn.pkl', 'wb')
pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()