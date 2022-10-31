#!/usr/bin/python3
import pandas as pd
import numpy as np
import sys
import io
from textwrap import indent
sys.path.insert(0, ".")

import utils
import CaseHandler as ch
import ModelHandler as mh
import gc

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
## for logging history
from keras.callbacks import CSVLogger
from tensorflow.python.keras.metrics import Metric
import json

import random

from sklearn.model_selection import KFold


class NeuralNetwork():
    
    def __init__(self):

        self.CASE               = ch.CaseHandler()

        self.df_features        = pd.read_csv(self.CASE.df_features_path, sep='\t', index_col=[0,1])
        self.df_labels          = pd.read_csv(self.CASE.df_labels_path, sep='\t', index_col=[0])


        self.trainingshape      = (self.CASE.trainingset_size,
                                   self.CASE.number_of_features,
                                   self.CASE.number_of_subfeatures)
        self.testshape          = (self.CASE.testset_size,
                                   self.CASE.number_of_features,
                                   self.CASE.number_of_subfeatures)

        self.train_prediction   = np.ones(( self.CASE.trainingset_size,
                                            self.CASE.number_of_labels))
        self.test_prediction    = np.ones(( self.CASE.testset_size,
                                            self.CASE.number_of_labels))    
        self.train_labels       = np.ones(( self.CASE.trainingset_size))
        self.test_labels        = np.ones(( self.CASE.testset_size))

        self.max_subfeatures    = np.ones((self.CASE.number_of_subfeatures))
        self.max_labels         = np.ones((self.CASE.number_of_labels))

        self.reindex_list_train = [i for i in range(0, self.CASE.trainingset_size)]
        self.reindex_list_test  = [i for i in range(0, self.CASE.testset_size)]
        
        self.model_summary      = ''


    def trainNetwork(self):

        if self.CASE.do_validation:
            self._validateModel()
        else:
            self._finalizeModel()


    def _finalizeModel(self):
        sample_order = [i for i in range(0, self.CASE.number_of_objects)]
        if self.CASE.shuffle_samples_pre:
            random.shuffle(sample_order)
        ### Prepare datasets
        ## Function includes:
        ## - shuffling sample order and/or point order
        self._prepareDatasets(sample_order, [])
        self._doFinalization()
        self._postProcessFinalzation()

    def _validateModel(self):
        ### get split indices for trainingset and testset
        dataSplit   = self._getDataSplittingIndices()
        ### train model for each fold
        fold_idx    = 0
        for train_idx, test_idx in dataSplit:
            ### Prepare datasets
            ## Function includes:
            ## - splitting into training- and testset
            ## - shuffling sample order and/or point order
            self._prepareDatasets(train_idx, test_idx)
            ## do training for each repetition
            for pass_idx in range(0, self.CASE.number_of_passes):
                self._doValidation(fold_idx, pass_idx)
                ## process and save predictions
                self._postProcessValidation(fold_idx, pass_idx)

            del self.train_dataset, self.test_dataset
            del self.train_labels, self.train_prediction
            gc.collect()

            fold_idx += 1
        
        self._saveModelSummaryToFile()
        gc.collect()

    def _saveModelSummaryToFile(self):
        ### save model summary to file
        path    = self.CASE.training_dict['saveData']['modelData']['directory'] \
                + self.CASE.training_dict['saveData']['modelData']['summary']
        with open(path, 'w') as f:
            f.write(self.model_summary)
        f.close()

    def _getDataSplittingIndices(self):
    
        if self.CASE.validation_type == 'k-folding':
                kfold       = KFold(n_splits = self.CASE.k, shuffle=self.CASE.shuffle_samples_pre)
                dataSplit   = kfold.split(np.zeros((self.CASE.number_of_objects)))
        elif self.CASE.validation_type == 'holdout':
                sample_order = [i for i in range(0, self.CASE.number_of_objects)]
                if self.CASE.shuffle_samples_pre:
                    random.shuffle(sample_order)
                dataSplit   = [[i for i in sample_order[0:self.CASE.trainingset_size]], 
                                [i for i in sample_order[self.CASE.trainingset_size:self.CASE.trainingset_size+self.CASE.testset_size]]]

        return dataSplit

    
    def _getVerticeIndices(self, train_idx, test_idx):
        ### get list of vertice indices
        random_order    = [i for i in range(0, self.CASE.number_of_features)]
        vertice_train_indices   = []
        vertice_test_indices    = []
        for sample_idx in range(0, self.CASE.trainingset_size):
            random.shuffle(random_order)
            vertice_train_indices   = vertice_train_indices + random_order
            if sample_idx < self.CASE.testset_size:
                random.shuffle(random_order)
                vertice_test_indices    = vertice_test_indices + random_order
        ### trainingset
        arrays                  = [[i for i in train_idx for _ in range(0, self.CASE.number_of_features)], \
                                   vertice_train_indices]
        tuples                  = list(zip(*arrays))
        feature_train_indices   = pd.MultiIndex.from_tuples(tuples, names=['objIdx', 'pntIdx'])
        ### testset
        arrays                  = [[i for i in test_idx for _ in range(0, self.CASE.number_of_features)], \
                                   vertice_test_indices]
        tuples                  = list(zip(*arrays))
        feature_test_indices    = pd.MultiIndex.from_tuples(tuples, names=['objIdx', 'pntIdx'])

        return feature_train_indices, feature_test_indices
        
        
    def _prepareDatasets(self, train_idx, test_idx):
        ### create datset for training and test data and do further modifications
        ## on data
        ## 1.) shuffle sample order before splitting
        if self.CASE.shuffle_samples_post:
            random.shuffle(train_idx)
            random.shuffle(test_idx)

        self.reindex_list_train = train_idx
        self.reindex_list_test  = test_idx
        
        ## 2.) split into training- and testset dataframes
        df_features_train       = self.df_features.loc[train_idx]
        df_features_test        = self.df_features.loc[test_idx]

        df_labels_train         = self.df_labels.loc[train_idx]
        df_labels_test          = self.df_labels.loc[test_idx]

        ## 3.) shuffle vertice order
        if self.CASE.shuffle_vertices:
            #self.train_dataset  = self.train_dataset.map(self._shuffleVertices)
            #self.test_dataset   = self.test_dataset.map(self._shuffleVertices)
            feature_train_indices, feature_test_indices = self._getVerticeIndices(train_idx, test_idx)
            df_features_train   = df_features_train.reindex(feature_train_indices)
            df_features_test    = df_features_test.reindex(feature_test_indices)
    
        ### get max values
        self.max_subfeatures    = df_features_train.max().values
        self.max_labels         = df_labels_train.max().values

        ### 3.) create datasets
        self.train_dataset      = tf.data.Dataset.from_tensor_slices(([df_features_train.values.reshape(self.trainingshape)], 
                                                                      [df_labels_train.values]))
        self.test_dataset       = tf.data.Dataset.from_tensor_slices(([df_features_test.values.reshape(self.testshape)], 
                                                                      [df_labels_test.values]))
        del df_features_train, df_features_test, df_labels_train, df_labels_test
        gc.collect()

        ### 5.) normalize dataframes
        if self.CASE.do_normalization:
            self.train_dataset  = self.train_dataset.map(self._normalize)
            if self.CASE.do_validation:
                self.test_dataset   = self.test_dataset.map(self._normalize)
    
        ### add batch size information
        self.train_dataset.batch(self.CASE.batch_size)
        if self.CASE.do_validation:
            self.test_dataset.batch(self.CASE.batch_size)

    def _shuffleVertices(self, samples, label):
        # get random indices
        sample_size     = samples.shape[0]
        reordered_sample = [] 
        for sample_idx in range(0, sample_size):
            reordered_sample.append(tf.random.shuffle(samples[sample_idx]))

        return reordered_sample, label

    def _normalize(self, samples, label):
        samples = samples/self.max_subfeatures
        label   = label/self.max_labels
        return samples, label


    def _doFinalization(self):
        print('==================================================================================================')
        print('==================================================================================================')
        print('FINALIZATION')

        ### create new keras neural network model
        model       = self._createNewModel()
        ### create callbacks for saving loss history
        ## define current path
        path        = self.CASE.training_dict['saveData']['predictionData']['directory']
        ## create current path
        utils.createDirectory(path)
        ## create current file
        filepath    = path + self.CASE.training_dict['saveData']['predictionData']['loss']
        csv_logger  = CSVLogger(filepath, separator="\t", append=False)
            
        ### fit model
        model.fit(self.train_dataset, 
                  epochs=self.CASE.epochs, 
                  callbacks=[csv_logger])
        ### save model
        path        = self.CASE.training_dict['saveData']['modelData']['directory']
        ## create current path
        utils.createDirectory(path)
        ## save
        #model.save(path)
        ### save model weights
        #filepath    = path + self.CASE.training_dict['saveData']['modelData']['weights']
        #model.save_weights(filepath)
        ### do predictions
        ## trainingset
        train_data                          = self.train_dataset.take(1)
        train_features, self.train_labels   = list(train_data)[0]
        train_features                      = np.array(train_features)
        self.train_labels                   = np.array(self.train_labels)
        ## run prediction
        self.train_prediction               = model.predict(train_features)

        del model, train_data, train_features
        gc.collect()



    def _doValidation(self, fold_idx, pass_idx):
        print('==================================================================================================')
        print('==================================================================================================')
        print('RUNTHROUGH')
        print('fold NO          : ', fold_idx)
        print('pass NO          : ', pass_idx)
        print('trainingset size :', self.CASE.trainingset_size)
        print('testset size     :', self.CASE.testset_size)

        ### create new keras neural network model
        model       = self._createNewModel()
        ### create callbacks for saving loss history
        ## define current path
        path        = self.CASE.training_dict['saveData']['predictionData']['directory'] \
                    + 'fold_%i/' %(fold_idx) \
                    + 'pass_%i/' %(pass_idx) 
        ## create current path
        utils.createDirectory(path)
        ## create current file
        filepath    = path + self.CASE.training_dict['saveData']['predictionData']['loss']
        csv_logger  = CSVLogger(filepath, separator="\t", append=False)
        ### fit model
        model.fit(self.train_dataset, 
                  epochs=self.CASE.epochs, 
                  validation_data=self.test_dataset,
                  callbacks=[csv_logger])
        ### save model
        path        = self.CASE.training_dict['saveData']['modelData']['directory'] \
                    + 'fold_%i/' %(fold_idx) \
                    + 'pass_%i/' %(pass_idx)
        ## create current path
        utils.createDirectory(path)
        ### save model weights
        filepath = path + self.CASE.training_dict['saveData']['modelData']['weights']
        model.save_weights(filepath)
        ### do predictions
        ## trainingset
        train_data                          = self.train_dataset.take(1)
        train_features, self.train_labels   = list(train_data)[0]
        train_features                      = np.array(train_features)
        self.train_labels                   = np.array(self.train_labels)
        ## testset
        test_data                           = self.test_dataset.take(1)
        test_features, self.test_labels     = list(test_data)[0]
        test_features                       = np.array(test_features)
        self.test_labels                    = np.array(self.test_labels)
        ## run prediction
        self.train_prediction               = model.predict(train_features)
        self.test_prediction                = model.predict(test_features)

        del model

    def _createNewModel(self):

        MH          = mh.geom2parNet()

        model       = MH.initModel()
        optimizer   = MH.optimizer
        loss        = MH.loss
        epochs      = MH.epochs

        model.compile(optimizer=optimizer, loss=loss)

        ### save model summary to string
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        self.model_summary = "\n".join(stringlist)
    
        return model

    def _postProcessFinalzation(self):
        ### rescale prediction
        if self.CASE.do_normalization:
            self.train_prediction   = self.train_prediction*self.max_labels
            self.train_labels       = self.train_labels*self.max_labels

        ### create arrays that contain prediction and true values
        trainset                = np.zeros((self.CASE.number_of_labels, 
                                            int(self.CASE.trainingset_size), 
                                            2))
        ### do metric calculations
        ##  metric array: [label_idx][metric_idx][set_idx]
        #   metric_idx = 0: RMSQ
        #   metric_idx = 1: MAPE
        #   metric_idx = 2: R SQUARED
        #   set_idx = 0: trainingset
        metrices            = np.zeros((self.CASE.number_of_labels, 3, 1))

        ### write to file
        for label_idx in range(0, self.CASE.number_of_labels):
            ### do metrics
            ## RMSE
            m = tf.keras.metrics.RootMeanSquaredError()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,0,0]             = m.result().numpy()
            ## MAPE
            m = tf.keras.metrics.MeanAbsolutePercentageError()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,1,0]             = m.result().numpy()
            ## R SQUARED
            m = tfa.metrics.RSquare()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,2,0]             = m.result().numpy()
            ## update result arrays
            trainset[label_idx,:,0]  = self.train_labels[:,label_idx]        
            trainset[label_idx,:,1]  = self.train_prediction[:,label_idx]                       

        ### write to file
        ## write results to file
        ## convert result arrays to dataframes
        arrays          = [[i for i in self.CASE.label_names for _ in range(self.CASE.number_of_labels)], \
                           ['trueVal', 'pred']*self.CASE.number_of_labels]
        tuples          = list(zip(*arrays))
        index           = pd.MultiIndex.from_tuples(tuples, names=['label', 'source'])

        df_trainingset  = pd.DataFrame(trainset.reshape(int(self.CASE.trainingset_size),2*self.CASE.number_of_labels), columns=index)
        df_trainingset.index.name = 'objIdx'

        ## save to file
        path            = self.CASE.prediction_path

        utils.createDirectory(path)
        df_trainingset.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['trainingset'],
                            sep='\t',
                            mode='w')
        ## sort dataframe by value
        df_trainingset.index    = self.reindex_list_train
        df_trainingset          = df_trainingset.sort_index()
        df_trainingset.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['trainingset'] + '_sorted',
                            sep='\t',
                            mode='w')

        ## convert metric array to dataframe
        arrays          = [i for i in [self.CASE.label_names]]
        tuples          = list(zip(*arrays))
        index           = pd.MultiIndex.from_tuples(tuples, names=['label'])
        metrices = np.array([metrices[:,i].reshape(self.CASE.number_of_labels) for i in range(0, 3)])
        df_metrices     = pd.DataFrame(metrices, index=['RMSE', 'MAPE','RSQUARE'], columns=index)
        df_metrices.index.name = 'metric'
        ## save metric dataframe to file
        df_metrices.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'],
                                     sep='\t',
                                     mode='w')
        print(df_metrices)

        ## write case parameters to file
        f = open(path + self.CASE.data_dict['dataPreparation']['writeParametersTo'], "w")
        f.write('// CASE DESCRIPTIVE PARAMETERS\n')
        parameter_list = [['trainingset size:', self.CASE.trainingset_size],
                          ['testset size:', self.CASE.testset_size]]
        tmp_list = ['Normalize data: ', self.CASE.do_normalization]
        parameter_list.append(tmp_list)
        for subfeature_idx, subfeature in enumerate(self.CASE.subfeature_names):
            tmp_list = ['subfeature #%i [%s] scale factor:' %(subfeature_idx, subfeature_idx), self.max_subfeatures[subfeature_idx]]
            parameter_list.append(tmp_list)
        for label_idx, label in enumerate(self.CASE.label_names):
            tmp_list = ['label      #%i [%s] scale factor:' %(label_idx, label), self.max_labels[label_idx]]
            parameter_list.append(tmp_list)

        for par_name, par in parameter_list:
            f.write("{:<40} {:<6}\n".format(par_name, str(par)))

        f.close()

                              
                                                  
    def _postProcessValidation(self, fold_idx, pass_idx):

        ### rescale prediction
        if self.CASE.do_normalization:
            self.train_prediction   = self.train_prediction*self.max_labels
            self.test_prediction    = self.test_prediction*self.max_labels
            self.train_labels       = self.train_labels*self.max_labels
            self.test_labels        = self.test_labels*self.max_labels

        ### create arrays that contain prediction and true values
        trainset                = np.zeros((int(self.CASE.trainingset_size),
                                            2*self.CASE.number_of_labels))
        testset                 = np.zeros((int(self.CASE.testset_size),
                                            2*self.CASE.number_of_labels))
                                                                                           
        trainset[:,0::2]  = self.train_labels
        trainset[:,1::2]  = self.train_prediction

        testset[:,0::2]  = self.test_labels
        testset[:,1::2]  = self.test_prediction

        ### do metric calculations 
        ##  metric array: [label_idx][metric_idx][set_idx]
        #   metric_idx = 0: MAE
        #   metric_idx = 1: MSE
        #   metric_idx = 2: MAPE
        #   metric_idx = 3: RMSE
        #   metric_idx = 4: R SQUARED
        #   set_idx = 0: trainingset
        #   set_idx = 1: testset
        metrices            = np.zeros((self.CASE.number_of_labels, 5, 2))

        ### write to file
        for label_idx in range(0, self.CASE.number_of_labels):
            ### do metrics
            ## MAE
            m = tf.keras.metrics.MeanAbsoluteError()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,0,0]             = m.result().numpy()
            m.update_state(self.test_labels[:,label_idx], self.test_prediction[:,label_idx])
            metrices[label_idx,0,1]             = m.result().numpy()
            ## MSE
            m = tf.keras.metrics.MeanSquaredError()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,1,0]             = m.result().numpy()
            m.update_state(self.test_labels[:,label_idx], self.test_prediction[:,label_idx])
            metrices[label_idx,1,1]             = m.result().numpy()
            ## MAPE
            m = tf.keras.metrics.MeanAbsolutePercentageError()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,2,0]             = m.result().numpy()
            m.update_state(self.test_labels[:,label_idx], self.test_prediction[:,label_idx])
            metrices[label_idx,2,1]             = m.result().numpy()
            ## RMSE
            m = tf.keras.metrics.RootMeanSquaredError()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,3,0]             = m.result().numpy()
            m.update_state(self.test_labels[:,label_idx], self.test_prediction[:,label_idx])
            metrices[label_idx,3,1]             = m.result().numpy()
            ## R SQUARED
            m = tfa.metrics.RSquare()
            m.update_state(self.train_labels[:,label_idx], self.train_prediction[:,label_idx])
            metrices[label_idx,4,0]             = m.result().numpy()
            m.update_state(self.test_labels[:,label_idx], self.test_prediction[:,label_idx])
            metrices[label_idx,4,1]             = m.result().numpy()

        ### write to file
        ## write results to file
        ## convert result arrays to dataframes

        arrays          = [[i for i in self.CASE.label_names for _ in range(0, 2)], \
                           ['trueVal', 'pred']*self.CASE.number_of_labels]
        tuples          = list(zip(*arrays))
        index           = pd.MultiIndex.from_tuples(tuples, names=['label', 'source'])
        df_trainingset  = pd.DataFrame(trainset, columns=index)

        df_trainingset.index.name = 'objIdx'
        df_testset      = pd.DataFrame(testset, columns=index)
        df_testset.index.name = 'objIdx'
        ## save to file
        path        = self.CASE.prediction_path + 'fold_%i/' %(fold_idx) + 'pass_%i/' %(pass_idx)

        utils.createDirectory(path)
        df_trainingset.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['trainingset'],
                            sep='\t',
                            mode='w')
        df_testset.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['testset'],
                            sep='\t',
                            mode='w')


        ## convert metric array to dataframe
        arrays          = [[i for i in self.CASE.label_names for _ in range(0, 2)], \
                           ['trainingset', 'testset']*self.CASE.number_of_labels]
        tuples          = list(zip(*arrays))
        index           = pd.MultiIndex.from_tuples(tuples, names=['label', 'set'])
        ## reshape metric format for fitting dataframe
        metrices        = np.array([metrices[:,i].reshape(2*self.CASE.number_of_labels) for i in range(0, 5)])
        df_metrices     = pd.DataFrame(metrices, index=['MAE', 'MSE', 'MAPE', 'RMSE', 'RSQUARE'], columns=index)
        df_metrices.index.name = 'metric'
        ## save metric dataframe to file
        df_metrices.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'],
                                     sep='\t',
                                     mode='w')

        print(df_metrices)
        ## write case parameters to file
        f = open(path + self.CASE.data_dict['dataPreparation']['writeParametersTo'], "w")
        f.write('// CASE DESCRIPTIVE PARAMETERS\n')
        parameter_list = [['trainingset size:', self.CASE.trainingset_size],
                          ['testset size:', self.CASE.testset_size]]
        tmp_list = ['Normalize data: ', self.CASE.do_normalization]
        parameter_list.append(tmp_list)
        for subfeature_idx, subfeature in enumerate(self.CASE.subfeature_names):
            tmp_list = ['subfeature #%i [%s] scale factor:' %(subfeature_idx, subfeature_idx), self.max_subfeatures[subfeature_idx]]
            parameter_list.append(tmp_list)
        for label_idx, label in enumerate(self.CASE.label_names):
            tmp_list = ['label      #%i [%s] scale factor:' %(label_idx, label), self.max_labels[label_idx]]
            parameter_list.append(tmp_list)

        for par_name, par in parameter_list:
            f.write("{:<40} {:<6}\n".format(par_name, str(par)))

        f.close()


    def clearMemory(self):
        del self.train_dataset, self.test_dataset

        
