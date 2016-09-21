import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import History

## Constants
spambase_data = "spambase.data"
test_split = 0.3 # split over original data
nfold = 5
epochs = 200
batch_size = 32
learning_rate = 0.01
decay = 1e-6
momentum = 0.1
hidden_nodes = 64


## Loading data
orig_data = pd.read_csv(spambase_data, sep=',')

X_orig = (orig_data.ix[:,0:56].values).astype('float32')
y_orig = (orig_data.ix[:,57].values).astype('float32')

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

## Data split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_orig, y_orig, test_size=test_split)

## Build model
input_dim = X_train.shape[1]

def create_model():
    _model = Sequential()
    _model.add(Dense(hidden_nodes, input_dim=input_dim, init='uniform'))
    _model.add(Activation('sigmoid'))

    _model.add(Dense(1, init='uniform'))
    _model.add(Activation('sigmoid'))

    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum)
    _model.compile(optimizer=sgd,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return _model

best_model = None
best_model_loss = 100.0

## K-fold cross validation
kf = KFold(len(y_train), n_folds=nfold, shuffle=True)
for train, validation in kf:
    model = create_model()

    ## Train model
    history = model.fit(X_train[train], y_train[train],
        nb_epoch=epochs,
        batch_size=batch_size,
        verbose=None)

    ## Validate model
    loss_and_metrics = model.evaluate(X_train[validation], y_train[validation],
        batch_size=batch_size,
        verbose=0)
    
    print("-----------")
    for i in range(len(model.metrics_names)):
        metric = model.metrics_names[i]
        score = loss_and_metrics[i]
        print("%s: %s" % (metric, score))
    
    ## Select best model
    if loss_and_metrics[0] < best_model_loss:
        best_model_loss = loss_and_metrics[0]
        best_model = model

print("-----------------  Best model re-training....")
## Train best model on whole train set
history = best_model.fit(X_train, y_train,
        nb_epoch=epochs,
        batch_size=batch_size,
        verbose=None)

## Evaluate model
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
for i in range(len(model.metrics_names)):
    metric = model.metrics_names[i]
    score = loss_and_metrics[i]
    print("%s: %s" % (metric, score))
