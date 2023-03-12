import glob
import os
import pickle as pickle

import pandas as pd
import polars as pl
import pycaret as py
import sweetviz as sv
import vaex as vs
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from pycaret.clustering import *

    """_summary_
    this_class_uses_pandas_vaex_polars
    pycaret_machine_learning
    pycaret_clustering
    lazy_predict_machine_learning
    A java interface using the java
    fx library is also available
    """
class DataImport:
    """_summary_
    processing_the_input_files
    for_the_pycaret_lazypredict
    machine_learning_allowing
    _for_three_type_file_intake
    """
def __init__(self, \
             input_path,\
             choice,hd5, \
             transit, \
             filename, \
                 dir):
    """_summary_

    Args:
        input_path (_str_): _description_
        in_this_user_provide_the_filename
        for_the_machine_learning
        choice (_str_): _description_
        in_this_all_the_methods_for_the
        data_selection_such_as_pandas_
        vaex_polars_hd5_are_presented
        for_scaling
        hd5 (_str_): _description_
        the_hd5_file_to_be_specifically
        for_analysis
        filename (_str_): _description_
        provide_the_file_name_for_the_
        summary_stats
    """
    self.input = input_path
    self.dir = os.chdir(dir)
    self.hd5 = hd5
    self.transit = transit
    self.dir.input = glob.glob(dir/*.csv")
    self.choice = ['pandas', 'vaex', 'polars', 'hd5','transit', 'dirpath']
    for i in self.choice:
        if i == pandas:
            self.data = pd.read_csv(self.input_path)
            self.file_data = open(self.filename,'rb+')
            self.file_data.write(self.data.describe)
            self.file_data.write(self.data.info)
            self.file_data.write(self.data.corr())
            self.file_data.write(self.data.columns)
            self.file_data.write(self.data.isna())
            self.file_data.write(self.data)
            self.file
            self.file_data.close()
        if i == vaex:
            self.data = vaex.open(self.input).to_pandas()
            self.file_data=open(self.filename,'rb+')
            self.file_data.write(self.data.count(*))
            self.file_data.write(self.data.mean)
            self.file_data.write(self.data.std)
            self.file_data.close()
        if i == polars:
            self.file_data=open(self.filename,'rb+')
            self.data = pl.scan_csv(self.input)
            self.file_data.close()
        if i == hd5:
            self.hd5 = vaex.open(self.hd5)
            self.file_data=open(self.filename,'rb+')
            self.file_data.write(self.data.count(*))
            self.file_data.write(self.data.count())
            self.file_data.close()
        if i == transit:
            self.file_data = vaex.from_pandas(self.transit)
            self.file_data=open(self.filename,'rb+')
            self.file_data.write(self.data.count(*))
            self.file_data.write(self.data.count())
            self.file_data.close()
        if i == dirpath and i == pandas:
            self.file_data = pd.concat(pd.read_csv(f) for f in self.dir.input)
            self.file_data = open(self.filename,'rb+')
            self.file_data.write(self.data.describe)
            self.file_data.write(self.data.info)
            self.file_data.write(self.data.corr())
            self.file_data.write(self.data.columns)
            self.file_data.write(self.data.isna())
            self.file_data.write(self.data)
        if i = dirpath and i = vaex:
           self.file_data = vaex.concat(vaex.openmany(f) for f in self.dir.input)
           self.file_data=open(self.filename,'rb+')
           self.file_data.write(self.data.count(*))
           self.file_data.write(self.data.count())
           self.file_data.write(self.data.mean)
           self.file_data.write(self.data.std)
           self.file_data.close()
        else:
            self.file_data.write('No_file_provided')
            self.file_data.close()
            print(f'the_summary_stats_for_the_data:{self.filename}')

def  pycaretHypertunePandas(self,\
                      filename,\
                      X, y,\
                      column:str,\
                      random: int, \
                      var, \
                      parameters,\
                      models_number,\
                      model, \
                      test):
        self.X = self.data.drop(column, axis=1)
        self.y = self.data[self.data[column]]
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        setup(data=self.data, target=self.var)
        de_tree=create_model('dt')
        tune_model(de_tree)
        tune_model(de_tree, n_iter=parameters)
        tune_model(de_tree, optimize='AUC')
        model_count=compare_models(n_select=models_number)
        top_models= [tune_model(i) for i in model_count]
        print(f'the_summary_information:{top_models}')
        cluster = setup(self.data, normalize=True)
        model = create_model('self.model')
        model_generate = assign_model(model)
        assign_model = pd.read_csv('self.test')
        assign_predictions = predict_model(model, data = assign_model)
        self.model_write = open(self.filename, 'rb')
        self.model_write = self.model_write.write(models,'_'.join(['model','pipeline']))
        self.model_write = self.model_write.write(top_models)
        self.model_write.close()
        return top_models

def  pycaretHypertunePolars(self,\
                      filename,\
                      X, y,\
                      column:str,\
                      random: int, \
                      var, \
                      parameters,\
                      models_number,\
                      model \,
                      test):
        self.X = self.data.select(pl.all().exclude([column]))
        self.y = self.data.select([column])
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        setup(data=self.file_data, target=self.var)
        de_tree=create_model('dt')
        tune_model(de_tree)
        tune_model(de_tree, n_iter=parameters)
        tune_model(de_tree, optimize='AUC')
        model_count=compare_models(n_select=models_number)
        top_models= [tune_model(i) for i in model_count]
        print(f'the_summary_information:{top_models}')
        cluster = setup(self.data, normalize=True)
        model = create_model('self.model')
        model_generate = assign_model(model)
        assign_model = pd.read_csv('self.test')
        assign_predictions = predict_model(model, data = assign_model)
        self.model_write = open(self.filename, 'rb')
        self.model_write = self.model_write.write(models, '_'.join(['model','pipeline']))
        self.model_write = self.model_write.write(top_models)
        self.model_write.close()
        return top_models

def lazyFitPandas(self, X, y, column:str,random_state: int):
        self.X = self.data.drop(column, axis=1)
        self.y = self.data[self.file_data[column]]
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random
        self.X_train, self.X_test, self.y_train, self.y_test =
        train_test_split(self.X, self.y, test_size=.5, random_state=self.random_state)
        estimate = LazyClassifier(
            verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = estimate.fit
        (self.X_train, self.X_test, self.y_train, self.y_test)
        compare_report = sweetviz.compare(
            [self.X_train, self.X], [self.X_test, self.y])
        compare_report.show_html('self.data.html', open_browser=false)
        compare_report.show_notebook()
        print(f'the_models_evaluated_are:{models}')
        self.model_write = open(self.filename, 'rb')
        self.model_write = self.model_write.write(models)
        self.model_write = self.model_write.write(predictions)
        self.model_write.close()
        return models

def lazyRegressionPandas(self, X, y,column,random: int):
        self.X = self.data.drop(column, axis=1)
        self.y = self.data[self.data[column]]
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train, self.X_test, self.y_train, self.y_test =
        train_test_split(self.X, self.y, test_size=.5, random_state=self.random)
        estimate = LazyRegressor(
            verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = estimate.fit(
            self.X_train, self.X_test, self.y_train, self.y_test)
        compare_report = sweetviz.compare(
            [self.X_train, self.X], [self.X_test, self.y])
        compare_report.show_html('self.data.html', open_browser=True)
        compare_report.show_notebook()
        self.model_write = open(self.filename, 'rb')
        self.model_write = self.model_write.write(models)
        self.model_write = self.model_write.write(predictions)
        self.model_write.close()
        return models

def lazyFitPolars(self, X, y, column:str,random_state: int):
        self.X = self.data.select(pl.all().exclude([column]))
        self.y = self.data.select([column])
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random
        self.X_train, self.X_test, self.y_train, self.y_test =
        train_test_split(self.X, self.y, test_size=.5, random_state=self.random_state)
        estimate = LazyClassifier(
            verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = estimate.fit
        (self.X_train, self.X_test, self.y_train, self.y_test)
        compare_report = sweetviz.compare(
            [self.X_train, self.X], [self.X_test, self.y])
        compare_report.show_html('self.data.html', open_browser=false)
        compare_report.show_notebook()
        print(f'the_models_evaluated_are:{models}')
        self.model_write = open(self.filename, 'rb')
        self.model_write = self.model_write.write(models)
        self.model_write = self.model_write.write(predictions)
        self.model_write.close()
        return models

def lazyRegressionPolars(self, X, y,column,random: int):
        self.X = self.data.select(pl.all().exclude([column]))
        self.y = self.data.select([column])
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train, self.X_test, self.y_train, self.y_test =
        train_test_split(self.X, self.y, test_size=.5, self.random_state=random)
        estimate = LazyRegressor(
            verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = estimate.fit(
            self.X_train, self.X_test, self.y_train, self.y_test)
        compare_report = sweetviz.compare(
            [self.X_train, self.X], [self.X_test, self.y])
        compare_report.show_html('self.data.html', open_browser=True)
        compare_report.show_notebook()
        self.model_write = open(self.filename, 'rb')
        self.model_write = self.model_write.write(models)
        self.model_write = self.model_write.write(predictions)
        self.model_write.close()
        return models
