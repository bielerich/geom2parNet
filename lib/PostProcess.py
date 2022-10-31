import sys, os
import re
# for turning dataframe.info into string
import io   

sys.path.insert(0, ".")
import utils
import CaseHandler as ch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
'''
CLASS DATAHANDLER
Read in .stl files, do operations and save data in feature and label matrices

'''

class PostProcess():

    def __init__(self):

        self.CASE           = ch.CaseHandler()

        self.log_filename   = 'log.trainNetwork'

        if self.CASE.validation_type    == 'holdout':
            self.n_folds    = 1
        elif self.CASE.validation_type  == 'k-folding':
            self.n_folds    = self.CASE.k

        self.df_trainingset = None
        self.df_testset     = None
        self.df_loss        = None
        self.metric         = None

        self.winning_model      = {}
        for label in self.CASE.label_names:
            self.winning_model[label] = [[np.array((10**5, 10**5)), './'],
                                         [np.array((10**5, 10**5)), './'],
                                         [np.array((0, 0)), './'],
                                         [np.array((10**5, 10**5)), './']]

        #self.winning_model      = np.zeros((self.CASE.number_of_labels, 4, 2, 1))
        self.collected_metrics  = np.zeros((self.n_folds,
                                            self.CASE.number_of_passes,
                                            6,
                                            self.CASE.number_of_labels*2))

        self.averaged_metrics   = np.zeros((self.CASE.number_of_labels,
                                            4,
                                            2))

    def evaluateData(self):

        if self.CASE.do_validation:
            for fold_idx in range(0, self.n_folds):
                for random_repeat_idx in range(0, self.CASE.number_of_passes):
                    prediction_path    = self.CASE.prediction_path \
                                       + 'fold_%i/' %(fold_idx) \
                                       + 'pass_%i/' %(random_repeat_idx)
                    self._readData(prediction_path)
                    self._doEvaluation(fold_idx, random_repeat_idx, prediction_path)

            self._averageMetrics()
            self._saveWinningModel()
        else:
            pass


    def _readData(self, path):
        self.df_trainingset = pd.read_csv(path + self.CASE.training_dict['saveData']['predictionData']['trainingset'],
                                          sep='\t',
                                          header=[0, 1],
                                          index_col = [0])
        self.df_testset     = pd.read_csv(path + self.CASE.training_dict['saveData']['predictionData']['testset'],
                                          sep='\t',
                                          header=[0, 1],
                                          index_col = [0])
        self.df_loss        = pd.read_csv(path + self.CASE.training_dict['saveData']['predictionData']['loss'],
                                          sep='\t', 
                                          index_col = [0])
        self.df_metric      = pd.read_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'],
                                          sep='\t',
                                          header=[0,1],
                                          index_col = [0])

    def _readStatisticData(self, path):
        self.df_trainingset_dev = pd.read_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['trainingset'],
                                          sep='\t',
                                          header=[0, 1],
                                          index_col = [0])
        self.df_testset_dev     = pd.read_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['testset'],
                                          sep='\t',
                                          header=[0, 1],
                                          index_col = [0])
         

    def _averageMetrics(self):
        ### average loss and errors over passes for each fold folder
        mean_skill  = np.mean(self.collected_metrics, axis=2)
        std         = np.std(self.collected_metrics, axis=2)
        std_error   = std/(sqrt(self.n_folds))

        iterables = [[label for label in self.CASE.label_names], ['trainingset', 'testset']]
        columns   = pd.MultiIndex.from_product(iterables, names=['label', 'set'])
        iterables = [["RMSE", "MAPE", "R2", "LOSS"],['mean', 'std', 'stdError']]
        index     = pd.MultiIndex.from_product(iterables, names=['error', 'metric'])

        for fold_idx in range(0, self.n_folds):
            mean_skill_fold     = mean_skill[0,fold_idx]
            std_fold            = std[0,fold_idx]
            std_error_fold      = std_error[0,fold_idx]
            df_averaged_metrics = pd.DataFrame([],index=index, columns=columns)
            df_averaged_metrics.loc[["RMSE", "MAPE", "R2", "LOSS"],'mean',:]       = mean_skill_fold
            df_averaged_metrics.loc[["RMSE", "MAPE", "R2", "LOSS"],'std',:]        = std_fold
            df_averaged_metrics.loc[["RMSE", "MAPE", "R2", "LOSS"],'stdError',:]   = std_error_fold

            path = self.CASE.prediction_path + 'fold_%i/' %(fold_idx)
            df_averaged_metrics.to_csv( path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'],
                                        sep='\t',
                                        mode='w')

        ### average loss and errors over fold folders
        self.collected_metrics  = self.collected_metrics.reshape(self.n_folds*self.CASE.number_of_passes,6,2*self.CASE.number_of_labels)
        mean_skill              = np.mean(self.collected_metrics, axis=1)[0]
        std                     = np.std(self.collected_metrics, axis=1)[0]
        std_error               = std/(sqrt(self.n_folds))

        df_averaged_metrics = pd.DataFrame([],index=index, columns=columns)
        df_averaged_metrics.loc[["RMSE", "MAPE", "R2", "LOSS"],'mean',:]       = mean_skill
        df_averaged_metrics.loc[["RMSE", "MAPE", "R2", "LOSS"],'std',:]        = std
        df_averaged_metrics.loc[["RMSE", "MAPE", "R2", "LOSS"],'stdError',:]   = std_error
        path = self.CASE.prediction_path
        df_averaged_metrics.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'],
                                   sep='\t',
                                   mode='w')
        ### build collected_metrics dataframe
        iterables          = [[i for i in self.CASE.label_names], \
                           ['trainingset', 'testset']]
        columns   = pd.MultiIndex.from_product(iterables, names=['label', 'set'])
        iterables = [[i for i in range(0,self.n_folds)],[i for i in range(0,self.CASE.number_of_passes)],["MAE", "MSE", "MAPE", "RMSE", "R2", "LOSS"]]
        index     = pd.MultiIndex.from_product(iterables, names=['fold', 'pass','metric'])
        self.df_collected_metrics   = pd.DataFrame(self.collected_metrics.reshape(self.n_folds*self.CASE.number_of_passes*6,
                                                   2*self.CASE.number_of_labels), index=index, columns=columns)
        ## save to file
        self.df_collected_metrics.to_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'] + '_collected',
                                         sep='\t',
                                         mode='w')
        self.df_collected_metrics.xs('LOSS', level=2, axis=0, drop_level=True).to_csv(path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'] + '_lloss',
                                         sep='\t',
                                         mode='w')


    def _doEvaluation(self, fold_idx, random_repeat_idx, prediction_path):
        ### dataframes for absolute and relative deviations between prediction and true value
        iterables = [[label for label in self.CASE.label_names], ['abs', 'rel', '|rel|']]
        columns   = pd.MultiIndex.from_product(iterables, names=['label', 'metric'])
        df_trainingset_dev  = pd.DataFrame([], columns=columns)
        df_testset_dev      = pd.DataFrame([], columns=columns)

        ### dataframes for standard deviation and variance
        iterables       = [['std', 'var'], ["abs", "rel", "|rel|"]]
        index           = pd.MultiIndex.from_product(iterables, names=['measure', 'metric'])
        df_statistics   = pd.DataFrame([],index=index, columns=['trainingset', 'testset'])
        ### do calculations for derivation dataframes 
        for label_idx, label in enumerate(self.CASE.label_names):
            ### trainingset
            df_trainingset_dev[label,'abs'] = self.df_trainingset[label]['trueVal']\
                                            - self.df_trainingset[label]['pred']
            df_trainingset_dev[label,'rel'] = df_trainingset_dev[label,'abs'] \
                                            / self.df_trainingset[label]['trueVal']*100
            df_trainingset_dev[label,'|rel|'] = abs(df_trainingset_dev[label,'rel'])
            ### testset
            df_testset_dev[label,'abs'] = self.df_testset[label]['trueVal'] \
                                        - self.df_testset[label]['pred']
            df_testset_dev[label,'rel'] = df_testset_dev[label,'abs'] \
                                        / self.df_testset[label]['trueVal']*100
            df_testset_dev[label,'|rel|'] = abs(df_testset_dev[label,'rel'])
            ### fill collected_metrics array to later average metric loss data
            metric_values   = self.df_metric.iloc[0:5].values
            loss_values     = self.df_loss[-1:].values[0]
            # copy loss values self.CASE.number_of_labels times for compatibility with dataframe format
            loss_values     = np.array([loss_values for _ in range(0, self.CASE.number_of_labels)]).reshape(1, self.CASE.number_of_labels*2)
            collected_metrics   = np.concatenate((metric_values, loss_values))
            self.collected_metrics[fold_idx, random_repeat_idx, :] = collected_metrics
            ## check if model is winning one
            self._checkForWinningModel(label, metric_values, loss_values, prediction_path)
            ### adding last loss LLOSS and standard deviation to metric dataframe
            self.df_metric.loc['LLOSS',:] = loss_values
            trainingset_std             = df_trainingset_dev.std(axis=0,skipna = True).values[0:self.CASE.number_of_labels]
            testset_std                 = df_testset_dev.std(axis=0,skipna = True).values[0:self.CASE.number_of_labels]
            self.df_metric.loc['STD',:] = np.concatenate((trainingset_std, testset_std), axis=0)
            ### save datasets
            df_trainingset_dev.to_csv(prediction_path + self.CASE.training_dict['saveData']['predictionData']['statistics']['trainingset'],
                                      sep='\t',
                                      mode='w')
            df_testset_dev.to_csv(prediction_path + self.CASE.training_dict['saveData']['predictionData']['statistics']['testset'],
                                  sep='\t',
                                  mode='w')
            self.df_metric.to_csv(prediction_path + self.CASE.training_dict['saveData']['predictionData']['statistics']['metric'],
                                 sep='\t',
                                 mode='w')

    def _checkForWinningModel(self, label, metric_values, loss_values, prediction_path):
        ### check for metrics:
        #   metric_idx = 0: MAE
        #   metric_idx = 1: MSE
        #   metric_idx = 2: MAPE
        #   metric_idx = 3: RMSE
        #   metric_idx = 4: R SQUARED
        ##  R2: metric_idx = 2
        for metric_idx in range(0, 3):
            norm_model           = np.linalg.norm(metric_values[metric_idx])
            norm_winning_model   = np.linalg.norm(self.winning_model[label][metric_idx][0])
            if metric_idx == 2: ## R2 to be topped
                if norm_model > norm_winning_model:
                    self.winning_model[label][metric_idx][0] = metric_values[metric_idx]
                    self.winning_model[label][metric_idx][1] = prediction_path
            else:
                if norm_model < norm_winning_model:
                    self.winning_model[label][metric_idx][0] = metric_values[metric_idx]
                    self.winning_model[label][metric_idx][1] = prediction_path

        ##  LOSS:    metric_idx = 3
        metric_idx              = 3
        norm_model           = np.linalg.norm(loss_values[0])
        norm_winning_model   = np.linalg.norm(self.winning_model[label][metric_idx][0])
        if norm_model < norm_winning_model:
            self.winning_model[label][metric_idx][0] = loss_values[0]
            self.winning_model[label][metric_idx][1] = prediction_path


    def _saveWinningModel(self):
        ### initialize an empty dataframe of best model choice
        iterables = [[label for label in self.CASE.label_names], ['trainingset', 'testset', 'path']]
        columns                     = pd.MultiIndex.from_product(iterables, names=['label', 'set'])
        df_winning_model         = pd.DataFrame([], index=["RMSE", "MAPE", "R2", "LOSS"], columns=columns)
        ### fill dataframe
        for label_idx, label in enumerate(self.CASE.label_names):
            metric_values = np.zeros((2, 4))
            metric_pathes = ['./','./','./','./']
            for metric_idx, metric in enumerate(self.winning_model[label]):
                metric_values[0, metric_idx] = metric[0][0]
                metric_values[1, metric_idx] = metric[0][1]
                metric_pathes[metric_idx]    = metric[1]
            df_winning_model[label,'trainingset']   = metric_values[0]
            df_winning_model[label,'testset']       = metric_values[1]
            df_winning_model[label,'path']          = metric_pathes
        ### write dataframe to file
        path = self.CASE.training_dict['saveData']['predictionData']['directory'] \
             + self.CASE.training_dict['saveData']['predictionData']['winning']
        df_winning_model.to_csv(path,sep='\t',mode='w')


    def _getSortOrderings(self,label):

            ### get index values for dataframe in ascending order of label value
            sortValIndex_trainingset= np.argsort(self.df_trainingset[(label, 'trueVal')].to_numpy())
            sortValIndex_testset   = np.argsort(self.df_testset[(label, 'trueVal')].to_numpy())

            distance_trainingset        = np.zeros(self.CASE.trainingset_size)
            distance_testset            = np.zeros(self.CASE.testset_size)
            for objIdx, obj in enumerate(self.df_trainingset[(label, 'trueVal')]):
                idx_1       = [objIdx-1 if objIdx-1>=0 else objIdx][0]
                idx_2       = [objIdx+1 if objIdx+1<self.CASE.trainingset_size else objIdx][0]
                neighbour_1 = self.df_trainingset[(label, 'trueVal')][idx_1] 
                neighbour_2 = self.df_trainingset[(label, 'trueVal')][idx_2]
                distance_1  = abs((obj-neighbour_1)/(obj))
                distance_2  = abs((obj-neighbour_2)/(obj))
                distance_trainingset[objIdx]  = max([distance_1, distance_2])
            for objIdx, obj in enumerate(self.df_testset[(label, 'trueVal')]):
                idx_1       = [objIdx-1 if objIdx-1>=0 else objIdx][0]
                idx_2       = [objIdx+1 if objIdx+1<self.CASE.testset_size else objIdx][0]
                neighbour_1 = self.df_testset[(label, 'trueVal')][idx_1] 
                neighbour_2 = self.df_testset[(label, 'trueVal')][idx_2]
                distance_1  = abs((obj-neighbour_1)/(obj))
                distance_2  = abs((obj-neighbour_2)/(obj))
                distance_testset[objIdx]= max([distance_1, distance_2])

            sortDistIndex_trainingset   = np.argsort(distance_trainingset)
            sortDistIndex_testset       = np.argsort(distance_testset)

            return sortValIndex_trainingset, sortValIndex_testset, sortDistIndex_trainingset, sortDistIndex_testset


    def plotData(self):
        utils.createDirectory(self.CASE.prediction_path  + 'Plots/')
        ### plot averaged data of each fold and pass run
        ## MAE
        plot = self.df_collected_metrics.iloc[self.df_collected_metrics.index.get_level_values(2)=='MAE',:].plot.line(style=['bs-'])
        plot.set_xlabel("repeat idx (fold idx, pass idx)")
        plot.set_ylabel("RMSE")
        path = self.CASE.prediction_path  + 'Plots/MAE.png'
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)
        ## MSE
        plot = self.df_collected_metrics.iloc[self.df_collected_metrics.index.get_level_values(2)=='MSE',:].plot.line(style=['bs-'])
        plot.set_xlabel("repeat idx (fold idx, pass idx)")
        plot.set_ylabel("RMSE")
        path = self.CASE.prediction_path  + 'Plots/MSE.png'
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)
        ## MAPE
        plot = self.df_collected_metrics.iloc[self.df_collected_metrics.index.get_level_values(2)=='MAPE',:].plot.line(style=['bs-'])
        plot.set_xlabel("repeat idx (fold idx, pass idx)")
        plot.set_ylabel("MAPE")
        path = self.CASE.prediction_path  + 'Plots/MAPE.png' 
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)
        ## RMSE
        plot = self.df_collected_metrics.iloc[self.df_collected_metrics.index.get_level_values(2)=='RMSE',:].plot.line(style=['bs-'])
        plot.set_xlabel("repeat idx (fold idx, pass idx)")
        plot.set_ylabel("RMSE")
        path = self.CASE.prediction_path  + 'Plots/RMSE.png'
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)
        ## R2SQUARE
        plot = self.df_collected_metrics.iloc[self.df_collected_metrics.index.get_level_values(2)=='R2',:].plot.line(style=['bs-'])
        plot.set_xlabel("repeat idx (fold idx, pass idx)")
        plot.set_ylabel("R2")
        path = self.CASE.prediction_path  + 'Plots/R2.png'
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)
        ## LLOSS
        plot = self.df_collected_metrics.iloc[self.df_collected_metrics.index.get_level_values(2)=='LOSS',0:2].plot.line(style=['bs-'])
        plot.set_xlabel("repeat idx (fold idx, pass idx)")
        plot.set_ylabel("LAST LOSS")
        path = self.CASE.prediction_path  + 'Plots/LLOSS.png'
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)
        ### plot model results of winners
        winner_paths    = []
        for winner in self.winning_model:
            if winner[1] not in winner_paths:
                winner_paths.append(winner[1])

    def _plotModelData(self, path):
        ### read data
        self._readData(path)
        self._readStatisticData(path)
        ### create Plots folder
        utils.createDirectory(path + 'Plots/')

        for label in self.CASE.label_names:
                ### get sorting 
                sortValIndex_trainingset, sortValIndex_testset, sortDistIndex_trainingset, sortDistIndex_testset = self._getSortOrderings(label)
                ## predictions
                plot = self.df_trainingset[label].plot.bar(y=['trueVal','pred'],style=['bs-','ro-'])
                plot.set_xlabel("stl index")
                plot.set_ylabel(label)
                filepath = path + 'Plots/trainingset_%s_over_objIdx.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                plot = self.df_testset[label].plot.bar(y=['trueVal','pred'],style=['bs-','ro-'])
                plot.set_xlabel("stl index")
                plot.set_ylabel(label)
                filepath = path + 'Plots/testset_%s_over_objIdx.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                ## deviation absolute
                plot = self.df_trainingset_dev[label].plot.bar(y=['abs'],style=['bs-'])
                plot.set_xlabel("stl object index")
                plot.set_ylabel("absolute deviation")
                filepath = path + 'Plots/trainingset_%s_over_objIdx_abs.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                plot = self.df_testset_dev[label].plot.bar(y=['abs'],style=['bs-'])
                plot.set_xlabel("stl object index")
                plot.set_ylabel("absolute deviation")
                filepath = path + 'Plots/testset_%s_over_objIdx_abs.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                ## relative deviation
                plot = self.df_trainingset_dev[label].plot.bar(y=['rel'],style=['bs-'])
                plot.set_xlabel("stl object index")
                plot.set_ylabel("relative deviation")
                filepath = path +  'Plots/trainingset_%s_over_objIdx_rel.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                plot = self.df_testset_dev[label].plot.bar(y=['rel'],style=['bs-'])
                plot.set_xlabel("stl object index")
                plot.set_ylabel("relative deviation")
                filepath = path + 'Plots/testset_%s_over_objIdx_rel.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                ## deviation absolute
                plot = self.df_trainingset_dev[label].iloc[sortValIndex_trainingset,:].plot.bar(y=['abs'],style=['bs-'])
                plot.set_xlabel("stl object index (%s in ascending order of value)" %(label))
                plot.set_ylabel("absolute deviation")
                filepath = path + 'Plots/trainingset_%s_over_objIdx_abs_sortedVal.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                plot = self.df_testset_dev[label].iloc[sortValIndex_testset,:].plot.bar(y=['abs'],style=['bs-'])
                plot.set_xlabel("stl object index (%s in ascending order of value)" %(label))
                plot.set_ylabel("absolute deviation")
                filepath = path + 'Plots/testset_%s_over_objIdx_abs_sortedVal.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                ## deviation absolute
                plot = self.df_trainingset_dev[label].iloc[sortDistIndex_trainingset,:].plot.bar(y=['abs'],style=['bs-'])
                plot.set_xlabel("stl object index (%s in ascending order of distance)" %(label))
                plot.set_ylabel("absolute deviation")
                filepath = path + 'Plots/trainingset_%s_over_objIdx_abs_sortedDist.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)
                plot = self.df_testset_dev[label].iloc[sortDistIndex_testset,:].plot.bar(y=['abs'],style=['bs-'])
                plot.set_xlabel("stl object index (%s in ascending order of distance)" %(label))
                plot.set_ylabel("absolute deviation")
                filepath = path + 'Plots/testset_%s_over_objIdx_abs_sortedDist.png' %(label)
                fig = plot.get_figure()
                plt.savefig(filepath)
                plt.close(fig)

        ## losses
        plot = self.df_loss.plot.line(y=['loss', 'val_loss'],style=['bs-'])
        plot.set_xlabel("epochs")
        plot.set_ylabel("losses")
        path = path + 'Plots/losses.png'
        fig = plot.get_figure()
        plt.savefig(path)
        plt.close(fig)


       
