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

    parser.add_argument('model', action='store', type=str,
                        help='one of: "3ch-CNN", "CNN", "FCN", "BDT"')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # -- check that model makes sense:
    if args.model not in ['3ch-CNN', 'CNN', 'FCN', 'BDT']:
        raise ValueError("The model type needs to be one of '3ch-CNN', 'CNN', 'FCN', 'BDT'.")

    # -- load data:
    data = h5py.File(args.train_data)
    images = np.expand_dims(data['all_events']['hist'][:args.nb_events], -1)
    labels = data['all_events']['y'][:args.nb_events]
    weights = data['all_events']['weight'][:args.nb_events]
    weights = np.log(weights+1)
    #weights = weights**0.1                                                                                                                                                       
    val = h5py.File(args.val_data)
    images_val = np.expand_dims(val['all_events']['hist'][:args.nb_events], -1)
    labels_val = val['all_events']['y'][:args.nb_events]
    weights_val = val['all_events']['weight'][:args.nb_events]  

    # -- output file names
    model_weights = 'model_weights_weighted01_' + args.model + '.h5'
    predictions_file = 'prediction_nn_weighted01_' + args.model + '.npy'

    if args.model == 'BDT':
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV
        base_clf = GradientBoostingClassifier(verbose=2)
        parameters = {
            'n_estimators': [50, 100],
            'max_depth':[3, 5]
        }
        clf = GridSearchCV(base_clf, parameters, n_jobs=4, fit_params={'sample_weight': weights})
        clf.fit(images.reshape(images.shape[0], -1), labels)
        yhat = clf.predict_proba(images_val.reshape(images_val.shape[0], -1))
        np.save(predictions_file, yhat)

    elif 'CNN' in args.model:
        from keras.layers import (Input, Conv2D, LeakyReLU,
            BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten)
        from keras.models import Model
        from keras.callbacks import EarlyStopping, ModelCheckpoint

        if args.model == '3ch-CNN':
            def add_channels(_images, _data):
                layer_em = np.expand_dims(_data['all_events']['histEM'][:args.nb_events], -1)
                layer_track = np.expand_dims(_data['all_events']['histtrack'][:args.nb_events], -1)
                layer_em = layer_em / layer_em.max()
                layer_track = layer_track / layer_track.max()
                return np.concatenate(
                    (np.concatenate(
                        (_images, layer_em), axis=-1
                    ), layer_track), axis=-1
                )

            images = add_channels(images, data)
            images_val = add_channels(images_val, val)

        x = Input(shape=(images.shape[1], images.shape[2], images.shape[3]))
        h = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(x)
        #    h = BatchNormalization()(h)                                                                                                                                                     
        #    h = MaxPooling2D(pool_size=(2, 2))(h)                                                                                                                                           
        #    h = Dropout(0.5)(h)                  
        h = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        #    h = BatchNormalization()(h)                                                                                                                                                     
        #    h = MaxPooling2D(pool_size=(2, 2))(h)                                                                                                                                           
        #    h = Dropout(0.5)(h)                  
        h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
        #    h = BatchNormalization()(h)                                                                                                                                                     
        #    h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        #    h = BatchNormalization()(h)          
        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        #    h = Dropout(0.5)(h)
        y = Dense(1, activation='sigmoid')(h)
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
        try:
            model.load_weights(model_weights)
            print 'Weights loaded from ' + model_weights
        except IOError:
            print 'No pre-trained weights found'
        try:
            model.fit(images, labels,
                      batch_size=args.batch_size,
                      sample_weight=weights,
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
        yhat = model.predict(images_val, verbose=1, batch_size=args.batch_size)
       # score = model.evaluate(images_val, labels_val, sample_weight=weights_val, verbose=1)
       # print 'Validation loss:', score[0]
       # print 'Validation accuracy:', score[1]
        np.save(predictions_file, yhat)

    elif args.model == 'FCN':
        from keras.layers import (Input, Conv2D, LeakyReLU,
            BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten)
        from keras.models import Model
        from keras.callbacks import EarlyStopping, ModelCheckpoint

        x = Input(shape=(images.shape[1] * images.shape[2], ))        
        #h = Flatten()(x)
        h = Dense(2048, activation='relu')(x)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        h = Dense(2048, activation='relu')(h)
        h = Dropout(0.5)(h)
        y = Dense(1, activation='sigmoid')(h)
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

        try:
            model.load_weights(model_weights)
            print 'Weights loaded from ' + model_weights
        except IOError:
            print 'No pre-trained weights found'

        try:
            model.fit(images.reshape(images.shape[0], -1), labels,
                      batch_size=args.batch_size,
                      sample_weight=weights,
                      epochs=args.nb_epochs,
                      verbose=1,
                      callbacks = [
                        EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                        ModelCheckpoint(model_weights,
                            monitor='val_loss', verbose=True, save_best_only=True)
                      ],
                      validation_data=(images_val.reshape(images_val.shape[0], -1), labels_val, weights_val)
            )
        except KeyboardInterrupt:
            print 'Training finished early'

        model.load_weights(model_weights)
        yhat = model.predict(images_val.reshape(images_val.shape[0], -1), verbose=1, batch_size=args.batch_size)
        # score = model.evaluate(images_val, labels_val, sample_weight=weights_val, verbose=1)
        # print 'Validation loss:', score[0]
        # print 'Validation accuracy:', score[1]
        np.save(predictions_file, yhat)
'''
    elif args.model == '3DCNN':
        from keras.layers import (Input, Conv3D, Dropout, ZeroPadding3D, BatchNormalization, 
                          AveragePooling3D, Flatten, Dense)
        from keras.layers.advanced_activations import LeakyReLU
        from keras.models import Model
        from keras.callbacks import EarlyStopping, ModelCheckpoint

#        raise NotImplementedError('3DCNN method not yet implemented')
        layer_em = np.expand_dims(data['all_events']['histEM'][:args.nb_events], -1)
        layer_track = np.expand_dims(data['all_events']['histtrack'][:args.nb_events], -1)
        images = np.concatenate(
            (np.concatenate(
                (images, layer_em), axis=-1
            ), layer_track), axis=-1 
        )
        x = Input(shape=images.shape)
        h = Conv3D(64, (3, 3, 3), padding='same')(x)
        h = LeakyReLU()(h)
        h = Dropout(0.2)(h)
        
        h = ZeroPadding3D((2, 2, 2))(h)
        h = Conv3D(128, (3, 3, 3), padding='valid')(h)
        h = LeakyReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)
        
        h = ZeroPadding3D((2, 2, 2))(h)
        h = Conv3D(256, (3, 3, 3), padding='valid')(h)
        h = LeakyReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)
        
        h = ZeroPadding3D((1, 1, 1))(h)
        h = Conv3D(512, (3, 3, 3), padding='valid')(h)
        h = LeakyReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)
        h = AveragePooling3D((2, 2, 2))(h)
        h = Flatten()(h)
        y = Dense(1, activation='sigmoid')(h)
        
        model = Model(x, y)

        model.summary()
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        try:
            model.load_weights(model_weights)
            print 'Weights loaded from ' + model_weights
        except IOError:
            print 'No pre-trained weights found'
        try:
            model.fit(images, labels,
                      batch_size=args.batch_size,
                      sample_weight=weights,
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
        yhat = model.predict(images_val, verbose=1, batch_size=args.batch_size)
        # score = model.evaluate(images_val, labels_val, sample_weight=weights_val, verbose=1)
        # print 'Validation loss:', score[0]
        # print 'Validation accuracy:', score[1]
        np.save(predictions_file, yhat)
'''
