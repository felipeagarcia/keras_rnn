from RNN import RNN
import data_handler as data
import os
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 7
    batch_size = 131
    num_features = 9
    timesteps = 512
    rnn_size = 258
    max_len = 150
    learn_rate = 0.001
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    inputs, labels, test_inputs, test_labels = data.load_pampap2()
    shape = inputs.shape[1:]
    model = RNN(n_classes, shape)
    opt = Adam(lr=learn_rate)
    model.compile(opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(inputs, labels, batch_size=batch_size,
              epochs=num_epochs, validation_split=0.2)
    model.evaluate(test_inputs, test_labels, steps=10)
