import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
from mlflow.tracking import MlflowClient
 
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
 
#TRACKING_SERVER_HOST = "ec2-34-201-119-193.compute-1.amazonaws.com"
 
# mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
# print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")
 
#mlflow.set_tracking_uri("file:///C:/Users/vikas.chavhan/Documents/MLFlow/newenv/mlruns")
# mlflow.set_tracking_uri("http://ec2-34-201-119-193.compute-1.amazonaws.com:5000/")
# Get argument from CammandMLFLOW_TRACKING_URI: "http://mlflow-webserver:5000"


 
parser = argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required = False, default = 0.7)
parser.add_argument("--l1_ratio",type=float,required = False, default = 0.9)
args = parser.parse_args()
 
# evalution function
 
def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual,pred)
    return rmse, mae, r2
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
 
    data = pd.read_csv(r"C:\Users\VJ\Documents\MLFLOW\newenv\data\wine-quality.csv")
   

    # split the data in train and test
 
    train, test = train_test_split(data)
 
    # The predicted column is quality which is scalar from 3 ,9
    train_x = train.drop(["quality"], axis =1)
    test_x = test.drop(["quality"], axis =1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
 
 
    alpha = args.alpha
    l1_ratio = args.l1_ratio
    exp = mlflow.set_experiment(experiment_name="test_experiment")
 
    with mlflow.start_run(experiment_id=exp.experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
 
        predicted_qualities = lr.predict(test_x)
 
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
 
        print("Elastic model (alpha={:f}, l1_ratio={:f}:".format(alpha,l1_ratio))
        print(" RMSE: %s" % rmse)
        print(" MAE: %s" % mae)
        print(" R2: %s" %r2)
 
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("r2",r2)
        mlflow.sklearn.log_model(lr,"mymodel")
