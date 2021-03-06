#!/usr/bin/env python3

from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


from graph import *
from utils import *


# Configure:

## Define parameters
NB_EPOCH = 300
PATIENCE = 30  # early stopping patience
VAL_RATE = 0.2

## Config filters
localpool = {'sym_norm':True}
chebyshev = {'max_degree': 2, 'sym_norm':True}

## Select similarity
SIMILARITY = cosine


# Train-test
## Get data and biuld model
## X:features  A:graph  y:labels
X_train, X_test, y_train, y_test = load_from_csv('train.csv', 'test.csv', normalized=True)
model = GCN.from_data(X_train, y_train, X_test, similarity=SIMILARITY, filter=localpool)

# A = make_adj(X_train, X_test, similarity=SIMILARITY)
# model = GCN.from_data(X_train, y_train, X_test, adj_matrix=A, filter=FILTER.localpool)

## Callbacks for EarlyStopping
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)

## Train
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01), weighted_metrics=['acc'])
model.fit(X_train, y_train, val_rate=VAL_RATE,
          epochs=NB_EPOCH, verbose=1,
          shuffle=False, callbacks=[es_callback])

## Evaluate model on the test data
eval_results = model.evaluate(X_test, y_test)
print('''
      ================
      Test loss: {}
      Test accuracy: {}
      =================
      '''.format(*eval_results))

y_pred = model.predict()
y_pred = decode_onehot(y_pred).ravel()
print('''
    target    predict
    -----------------
    ''')
for y, yp in zip(y_test, y_pred):
    print(f'{y}    {yp}')

np.savetxt('test-pred.csv', np.column_stack((y_test, y_pred)),fmt='%d',delimiter=',')
