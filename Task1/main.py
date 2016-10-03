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
nfold = 5
epochs = 2500
batch_size = 32
learning_rate = [0.5, 0.3, 0.25, 0.2, 0.1, 0.01, 0.001]
decay = 1e-4
momentum = 0
hidden_nodes = 15

np.random.seed(42)

## Loading data
orig_data = pd.read_csv(spambase_data, sep=',')

X_orig = (orig_data.ix[:,0:56].values).astype('float32')
y_orig = (orig_data.ix[:,57].values).astype('float32')

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

def create_model(lr):
    _model = Sequential()
    _model.add(Dense(hidden_nodes, input_dim=input_dim, init='uniform'))
    _model.add(Activation('sigmoid'))

    _model.add(Dense(1, init='uniform'))
    _model.add(Activation('sigmoid'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum)
    _model.compile(optimizer=sgd,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return _model

best_model = None
best_model_loss = 100.0
best_model_index = -1
index = 0

for lr in learning_rate:
    model = None
    avg_loss = 0
    total_loss = 0
    Wsave = None
    ## K-fold cross validation
    kf = StratifiedKFold(y_train, n_folds=nfold, shuffle=True)
    for train, validation in kf:
        model = None
        model = create_model(lr)
        if Wsave:
            model.set_weights(Wsave)
        Wsave = model.get_weights()

        ## Train model
        history = model.fit(X_train[train], y_train[train],
            nb_epoch=epochs,
            batch_size=batch_size,
            # callbacks=[early_stopping],
            validation_data=(X_train[validation], y_train[validation]),
            verbose=2)

        ## Validate model
        loss_and_metrics = model.evaluate(X_train[validation], y_train[validation],
            batch_size=batch_size,
            verbose=0)
        
        print("----------- Model %s KF " % index)
        for i in range(len(model.metrics_names)):
            metric = model.metrics_names[i]
            score = loss_and_metrics[i]
            print("%s: %s" % (metric, score))
        total_loss += loss_and_metrics[0]

    avg_loss = total_loss / nfold

    print("----------- Model %s: average loss: %s " % (index, avg_loss))
    ## Select best model
    if avg_loss < best_model_loss:
        best_model_loss = avg_loss
        # best_model = pickle.dumps(model)
        best_model_index = index
    index += 1

print("-----------------  Best model (index: %s) re-training...." % best_model_index)
## Train best model on whole train set
best_model = create_model(learning_rate[best_model_index])
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

## NOT SURE WHY IF TRAINED OVER FEW CASES, ACC DROPS TO 58%, EVEN THOUGH THE MODEL CHOSEN IS THE BEST (PERFORMS 90+% IN KF; EVEN WHEN DOING ALONE)

# f = open('model_best_learnLr_2000ep_15hdn_5skf.pkl', 'wb')
# pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()
# f = open('training_2500ep_0.3lr_sgd_5skf_20hdn.pkl', 'wb')
# pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()