from utils import textToSequence, breakInto, removeDiacritics, toTarget
from keras.layers import Input, Bidirectional, LSTM, Dropout, TimeDistributed, Dense, GRU
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

from keras.utils import Sequence

from keras.preprocessing.sequence import TimeseriesGenerator

from utils import single_class_accuracy

import numpy as np
import os


class ChunkedReader(Sequence):
    def __init__(self, file, timesteps, batchSize):
        self.dataset = open(file, 'r')
        self.batchSize = batchSize
        self.timesteps = timesteps

    @property
    def chunkSize(self):
        # The chunkSize (the amount of bytes we need to read) should
        # be equal to the size of our batch times the number of timesteps
        # 
        # Since one timestep equals one character, we can easily multiply that
        # by our batch size and get the number of bytes to read.
        return self.batchSize * self.timesteps

    def __len__(self):
        stSize = os.fstat(self.dataset.fileno()).st_size
        return int(np.ceil(stSize / self.chunkSize)) # not sure if ok

    def __getitem__(self, idx):
        data = self.dataset.read(self.chunkSize)

        if data is '':
            raise StopIteration
        else:
            x = breakInto(textToSequence(removeDiacritics(data)))
            x = np.reshape(x, x.shape + (1,))
            
            y = toTarget(data)

            return (x, y)

class NeuralNetwork(object):
    def __init__(self, tsSize = 30, lstmSize = 128, gruSize = 64, dropout = 0.25):
        self.TIMESERIES_SIZE = tsSize

        inputs = Input(shape=(tsSize, 1))
        x = Bidirectional(LSTM(lstmSize, return_sequences=True))(inputs)
        x = Dropout(dropout)(x)
        x = Bidirectional(GRU(gruSize, return_sequences=True, activation='tanh'))(x)
        x = Dropout(dropout)(x)
        x = TimeDistributed(Dense(4, activation='softmax'))(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        
        optimizer = Adam(lr=0.0001)

        self.model.compile(optimizer, 
            'categorical_crossentropy', 
            metrics=['acc', 
                single_class_accuracy(0),  # ă, ț, ș
                single_class_accuracy(1),  # î
                single_class_accuracy(2)]) # â

    def epochFeedback(self, epoch, logs):
        for a in ["Langa casa mea e casa ta si e o casa foarte frumoasa.",
            "In casa era o casa si in casa aia era casa ta."]:
            print(self.predict(a))

    def predict(self, text = ''):
        X = breakInto(textToSequence(text))
        X = np.reshape(X, X.shape + (1,))
        pred = self.model.predict(X)
        pred = pred.reshape(-1, pred.shape[-1])
        
        out = []
        labels = [np.argmax(amax) for amax in pred[:len(text)]]

        for i, label in enumerate(labels):
            if label == 2 and text[i] in ['a', 'A']: 
                out.append('â')
            elif label == 1 and text[i] in ['i', 'I']: 
                out.append('î')
            elif label == 0 and text[i].lower() in ['a', 't', 's']: 
                if text[i].lower() == 'a': out.append('ă')
                elif text[i] == 't': out.append('ț') 
                elif text[i] == 's': out.append('ș') 
            else: out.append(text[i])
        return ''.join(out)

    def fit(self, checkpoint, train, test, epochs=3000, batch_size=50):
        self.model.fit_generator(
            generator = ChunkedReader(train, self.TIMESERIES_SIZE, batch_size),
            validation_data = ChunkedReader(test, self.TIMESERIES_SIZE, batch_size), 
            
            epochs=epochs,

            callbacks=[
                # Just a test sentence I use after each epoch, to see how things are going.
                LambdaCallback(self.epochFeedback), 

                # Save best model variations to disk. Filename argument is checkpoint.
                ModelCheckpoint(checkpoint, monitor='val_loss', verbose=1, save_best_only=True),
            ]
        )
