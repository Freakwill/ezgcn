在[GCN](https://github.com/zhouchunpong/GCN_Keras)上修改。更加接近scikit-learn用法。更加简单好用，几乎无需doc，看一个例子即可。


```python
#!/usr/bin/env python3

from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


from graph import *
from utils import *


# Define parameters
NB_EPOCH = 200
PATIENCE = 30  # early stopping patience

# config filters
FILTER.localpool = {'sym_norm':True}
FILTER.chebyshev = {'max_degree': 2, 'sym_norm':True}

# Get data
# X:features  A:graph  y:labels
X_train, X_test, y_train, y_test = load_from_csv('train.csv', 'test.csv', normalized=True)

# Compile model
A = make_adj(X_train, X_test)
model = GCN.from_data(X_train, y_train, X_test, adj_matrix=A, filter=FILTER.localpool)

# Callbacks for EarlyStopping
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=PATIENCE)

# Train
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01), weighted_metrics=['acc'])
model.fit(X_train, y_train,
          epochs=NB_EPOCH, verbose=1,
          shuffle=False, callbacks=[es_callback])

# Evaluate model on the test data
eval_results = model.evaluate(X_test, y_test)
print('''Test Done.
      Test loss: {}
      Test accuracy: {}'''.format(*eval_results))

preds = model.predict(graph, batch_size=A.shape[0])
np.savetxt('test-pred.csv', np.column_stack((y_test, np.argmax(preds[-y_test.shape[0]:], 1))),fmt='%d',delimiter=',')

```



