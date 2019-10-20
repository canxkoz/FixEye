import settings
import imports
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D

def cnn_model_old(input_shape_m):
    n_model = Sequential()

    n_model.add(Conv2D(6, (28, 28), padding='valid', input_shape=input_shape_m))
    n_model.add(Activation("sigmoid"))
    n_model.add(MaxPooling2D(pool_size=(2, 2)))

    n_model.add(Conv2D(16, (10, 10), padding='valid'))
    n_model.add(Activation("sigmoid"))
    n_model.add(MaxPooling2D(pool_size=(2, 2)))
    n_model.add(Flatten())

    n_model.add(Dense(120))
    n_model.add(Activation("sigmoid"))

    n_model.add(Dense(84))
    n_model.add(Activation("sigmoid"))

    n_model.add(Dense(settings.nb_classes))
    n_model.add(Activation('softmax'))

    return n_model

def cnn_model(input_shape):
    input = Input(shape=input_shape)
    model = Conv2D(32,(3,3),padding='same', activation='relu')(input)
    model = Conv2D(32, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.25)(model)
    
    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.3)(model)

    model = Flatten()(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.4)(model)
    output = Dense(settings.nb_classes, activation='softmax')(model)
    
    model = Model(inputs=input, outputs=output)

    return model
