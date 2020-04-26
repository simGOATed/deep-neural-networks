import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Model

def network(){
  a0 = Input(shape=(28, 28, 1))
  X = Conv2D(6, (3, 3), padding='same', activation='relu')(a0)
  X = MaxPool2D()(X)
  X = Conv2D(16, (3, 3), activation='relu')(X)
  X = MaxPool2D()(X)
  X = Flatten()(X)
  X = Dense(120, 'relu')(X)
  X = Dense(84, 'relu')(X)
  aL = Dense(10, 'softmax')(X)

  model = Model(a0, aL)
  return model
}

def train(model, x_train, y_train, epochs){
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
  model.fit(x_train, y_train, epochs)
}
