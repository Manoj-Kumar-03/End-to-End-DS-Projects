import os
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger_1 import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split Training and Test input")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'Linear Regression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'K-Neighbors Regressor':KNeighborsRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest Regressor':RandomForestRegressor(),
                'AdaBoost Regressor':AdaBoostRegressor()
            }

            params={
                "Linear Regression":{},
                'Lasso':{},
                'Ridge':{},
                'K-Neighbors Regressor':{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model Found")
            
            logging.info("Best Model Found on Training Data And Testing Data")

            save_object(
                file_path=self.trained_model_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test,predicted)

            return score

        except CustomException as e:
            raise CustomException(e,sys)
