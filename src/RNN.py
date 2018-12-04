from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import Sequential


class RNN(Sequential):
    def __init__(self, num_classes, input_shape, rnn_size=128, num_layers=2,
                 dropout_rate=0):
        super().__init__()
        self.num_classes = num_classes
        self.input_data_shape = input_shape
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self._create_model()

    def _create_model(self):
        self.add(LSTM(self.rnn_size,
                 input_shape=self.input_data_shape,
                 return_sequences=True))
        for layer in range(1, self.num_layers):
            self.add(LSTM(self.rnn_size))
        if self.dropout_rate > 0:
            self.add(Dropout(self.dropout_rate))
        self.add(Dense(self.num_classes, activation='softmax'))
