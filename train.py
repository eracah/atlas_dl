import h5py
import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(
        description='Run SUSY RPV training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')

    parser.add_argument('--nb-events', action='store', type=int, default=1000,
                        help='Number of events to train on.')

    parser.add_argument('--batch-size', action='store', type=int, default=256,
                        help='batch size per update')

    parser.add_argument('train_data', action='store', type=str,
                        help='path to HDF5 file to train on')

    parser.add_argument('val_data', action='store', type=str,
                        help='path to HDF5 file to validate on')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    data = h5py.File(args.train_data)
    images = np.expand_dims(data['all_events']['hist'][:args.nb_events], -1)
    labels = data['all_events']['y'][:args.nb_events]
#    weights = data['all_events']['weight'][:args.nb_events]
#    weights = np.log(np.log(weights))
    #print labels.mean()

    val = h5py.File(args.val_data)
    images_val = np.expand_dims(val['all_events']['hist'][:args.nb_events], -1)
    labels_val = val['all_events']['y'][:args.nb_events]
    weights_val = val['all_events']['weight'][:args.nb_events]
    #print labels_val.mean()

    from keras.layers import (Input, Conv2D, LeakyReLU, 
        BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten)

    x = Input(shape=(64, 64, 1))
    h = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
#    h = BatchNormalization()(h)
#    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Dropout(0.5)(h)

    h = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
#    h = BatchNormalization()(h)
#    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Dropout(0.5)(h)

    h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
#    h = BatchNormalization()(h)
#    h = MaxPooling2D(pool_size=(2, 2))(h)

    h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
#    h = BatchNormalization()(h)

    h = Flatten()(h)
    h = Dropout(0.5)(h)
    h = Dense(512, activation='relu')(h)
    h = Dropout(0.5)(h)
    y = Dense(1, activation='sigmoid')(h)

    from keras.models import Model
    model = Model(
        inputs = x,
        outputs = y
    )
    model.summary()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    model_weights = 'model_weights.h5'
    try:
        model.load_weights(model_weights)
    except IOError:
        print 'No pre-trained weights found'

    try:
        model.fit(images, labels,
                  batch_size=args.batch_size,
#                  sample_weight=weights,
                  epochs=args.nb_epochs,
                  verbose=1,
                  callbacks = [
                    EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                    ModelCheckpoint(model_weights,
                        monitor='val_loss', verbose=True, save_best_only=True)
                  ],
                  validation_data=(images_val, labels_val, weights_val)
        )
    except KeyboardInterrupt:
        print 'Training finished early'

    model.load_weights(model_weights)
    score = model.evaluate(images_val, labels_val, sample_weight=weights_val, verbose=1)
    print 'Validation loss:', score[0]
    print 'Validation accuracy:', score[1]



