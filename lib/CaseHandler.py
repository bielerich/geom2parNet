import sys
import pathlib
sys.path.insert(0, ".")
import DictHandler as dh
import utils

import numpy as np
import pandas as pd

from datetime import datetime
from string import Template

class CaseHandler():

    def __init__(self):

        ### Read dictionaries from pipeline/
        self.stl_dict               = {}
        self.data_dict              = {}
        self.training_dict          = {}
        self._readDicts()
        
        ### Define further case parameters
        ## STL parameters from stl dict
        self.stl_path               = self.stl_dict['directory']
        self.label_key_filepath     = self.stl_dict['labelKeyFile']['directory'] + self.stl_dict['labelKeyFile']['filename']
        self.stl_type               = self.stl_dict['objectParameters']['type']
        self.number_of_objects      = self.stl_dict['objectParameters']['numberOfObjects']
        self.control_parameters     = self.stl_dict['objectParameters']['controlParameters']

        ## data parameters
        self.subfeature_names       = list(self.data_dict['featureExtraction']['subFeatures'].keys())
        self.number_of_subfeatures  = len(self.subfeature_names)
        self.label_names            = list(self.stl_dict['objectParameters']['controlParameters'].keys())
        self.number_of_labels       = len(self.label_names)

        self.df_features_path       = self.data_dict['directory']+self.data_dict['featureExtraction']['filename']
        self.number_of_features     = self.data_dict['featureExtraction']['numberOfFeatures']
        self.df_labels_path         = self.data_dict['directory']+self.data_dict['labelExtraction']['filename']

        self.do_translation         = utils.isTrue(self.data_dict['objectTransformation']['translation'])
        self.do_rotation            = utils.isTrue(self.data_dict['objectTransformation']['rotation'])
        self.transformation_order   = self.data_dict['objectTransformation']['order']
        
        self.shuffle_samples_pre    = utils.isTrue(self.data_dict['dataPreparation']['shuffleSampleOrder']['preSplitting'])
        self.shuffle_samples_post   = utils.isTrue(self.data_dict['dataPreparation']['shuffleSampleOrder']['postSplitting'])
        self.shuffle_vertices       = utils.isTrue(self.data_dict['dataPreparation']['shuffleVerticeOrder'])

        self.do_normalization       = utils.isTrue(self.data_dict['dataPreparation']['normalize'])

        ## NN parameters        
        self.batch_size             = self.training_dict['model']['BATCH_SIZE']
        
        self.do_validation          = utils.isTrue(self.training_dict['doValidation'])
        self.epochs                 = self.training_dict['model']['EPOCHS']
        self.validation_type        = self.training_dict['modelValidation']['type']
        self.number_of_passes       = self.training_dict['modelValidation']['numberOfPasses']
        self.model_path             = self.training_dict['saveData']['modelData']['directory']
        self.prediction_path        = self.training_dict['saveData']['predictionData']['directory']

        if self.do_validation:
            self.k                  = self.training_dict['modelValidation']['k']
            self.folder_size        = int(self.stl_dict['objectParameters']['numberOfObjects']/self.k)
            self.trainingset_size   = int((self.k-1)*self.folder_size)
            self.testset_size       = int(self.folder_size)
            self.overall_passes     = int(self.k*self.number_of_passes)
                
        else:
            self.trainingset_size   = int(self.stl_dict['objectParameters']['numberOfObjects'])
            self.testset_size       = 0
            
    def _readDicts(self):
        Dict = dh.DictHandler()

        Dict.readStlDict()
        self.stl_dict = Dict.dict
        Dict.readDataDict()
        self.data_dict = Dict.dict
        Dict.readLearningDict()
        self.training_dict = Dict.dict

