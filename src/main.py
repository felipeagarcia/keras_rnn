from RNN import RNN
import data_handler as data
import os
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    num_epochs = 1000
    n_classes = 18
    batch_size = 85
    num_features = 113
    timesteps = 24
    max_len = 150
    learn_rate = 0.001
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    inputs, labels, test_inputs, test_labels = data.load_data()
    print(len(test_inputs))
    shape = inputs.shape[1:]
    model = RNN(n_classes, shape)
    opt = Adam(lr=learn_rate)
    model.compile(opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(inputs, labels, batch_size=batch_size,
              epochs=num_epochs, validation_split=0.2)
    acc, loss = model.evaluate(test_inputs, test_labels, batch_size=17)
    print("Model accuracy:", acc, ", model loss:", loss)
