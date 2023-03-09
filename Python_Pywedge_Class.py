import pywedge as pw
import pandas as pd
class Pywedge(DataImport,Features):
    def ___init__(self,input_path_train, input_path_test,model,target):
        self.input_path_train = input_path_train
        self.train = pd.read_csv(self.input_path_train)
        self.input_path_test = input_path_test
        self.test = pd.read_csv(self.input_path_test)
        self.target=str(target)
        self.model=model
        chart_train=pw.Pywedge_Charts(self.train,c=None,y=self.target)
        chart_test=pw.Pywedge_Charts(self.test,c=None,y=self.target)
        print(f'the_summary_data_describe:{pd.describe(self.train)}')
        print(f'the_summary_data_stats_on_your_data:{pd.info(self.train)}')
        print(f'the_summary_data_describe:{pd.describe(self.test)}')
        print(f'the_summary_data_stats_on_your_data:{pd.info(self.test)}')
        print(f'the_visualization_chart:{chart_train.make_charts()}')
        print(f'the_summary_data_describe:{pd.describe(self.test)}')
        print(f'the_summary_data_stats_on_your_data:{pd.info(self.test)}')
        print(f'the_summary_data_describe:{pd.describe(self.test)}')
        print(f'the_summary_data_stats_on_your_data:{pd.info(self.test)}')
        print(f'the_visualization_chart:{chart_test.make_charts()}')
        def trainingPywedge(self):
            for i in self.model:
                if i==Regression:
                    analyze_reg=pw.baseline_model(self.train,self.test,c=None,y=str(self.target), type='Regression')
                print(f'the_summary_of_regression_classification:{analyze_reg.Regression_summary()}')
                if i==Classification:
                    analyze_cls=pw.baseline_model(self.train,self.test,c=None,y=str(self.target), type='Classification')
                print(f'the_summary_of_regression_classification:{analyze_cls.Classification_summary()}')
            else:
                print(f'no_model_selected')
        
    def pywedgeHypertuneReg(self):
        hyper.reg=pw.Pywedge_HP(self.train, self.test, c=None, y=self.target)
        return hyper.HP_Tune_Regression()
    
     def pywedgeHypertuneClass(self):
        hyper.clas=pw.Pywedge_HP(self.train, self.test, c=None, y=self.target)
        return hyper.HP_Tune_Classification()
    
    
    