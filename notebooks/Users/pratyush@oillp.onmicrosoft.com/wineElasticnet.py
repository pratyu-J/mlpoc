# Databricks notebook source
user = dbutils.secrets.get("warehouse", "username")
password = dbutils.secrets.get("warehouse", "password")

options = {
  "sfUrl": "ep55011.central-india.azure.snowflakecomputing.com",
  "sfUser": user,
  "sfPassword": password,
  "sfDatabase": "WINESET",
  "sfSchema": "SCHEMA01",
  "sfWarehouse": "COMPUTE_WH"
}

# COMMAND ----------

import pandas as pd
 
from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid
 
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
 
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

data = spark.read.format('snowflake').options(**options).option('dbtable', 'WINE').load()
display(data)
def addIdColumn(dataframe, id_column_name):
    """Add id column to dataframe"""
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    """Rename columns to be compatible with Feature Store"""
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

renamed_df = renameColumns(data)
df = addIdColumn(renamed_df, 'wine_id')

# Drop target column ('quality') as it is not included in the feature table
features_df = df.drop('quality')
display(features_df)

spark.sql(f"CREATE DATABASE IF NOT EXISTS wine_db")
table_name = "wine_db.wine_ds_" + str(uuid.uuid4())[:6]
print(table_name)



# COMMAND ----------

#creating the feature table
fs = feature_store.FeatureStoreClient()
fs.create_feature_table(
    name = table_name,
    keys= ['wine_id'],
    features_df= features_df,
    schema = features_df.schema,
    description = "wine features")

# fs.create_table(
#     name = table_name,
#     primary_keys = ['wine_id'],
#     df = features_df,
#     schema = features_df.schema,
#     description = "wine features"
# )

#target df
target_df = df.select("wine_id", "quality", (10*rand()).alias("real_time_measurement"))
display(target_df)


#using FeatureLookup
def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
    model_target_lookups = [FeatureLookup(table_name = table_name+"_target", lookup_key=lookup_key)]
    
    training_set = fs.create_training_set(target_df, model_feature_lookups, label='quality', exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("quality", axis=1)
    Y = training_pd["quality"]
    xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2, random_state=0)
    return xtrain, xtest, ytrain, ytest, training_set

# Xtrain, Xtest, Ytrain, Ytest, training_set = load_data(table_name, "wine_id")
# print(Xtrain)

# COMMAND ----------

Xtrain, Xtest, Ytrain, Ytest, training_set = load_data(table_name, "wine_id")
print(Xtrain)


# COMMAND ----------

# experiment_name = '/Users/pratyush@oillp.onmicrosoft.com/mlflowPoc'
# mlflow.set_experiment(experiment_name)

# COMMAND ----------

# Working with mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import infer_signature
 
client = MlflowClient()
 
try:
    client.delete_registered_model("elnet_wine_model") # Delete the model if already created
except:
    None
    
#mlflow.sklearn.autolog(log_models=False)


def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    for x in [[.5,.9], [.2,.8], [.2,.9], [.4,.8], [.7,.7], [.8,.4]]:
        with mlflow.start_run() as run:

    #         rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
    #         rf.fit(X_train, y_train)
    #         y_pred = rf.predict(X_test)

            model = ElasticNet(alpha=x[0], l1_ratio=x[1])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(y_pred)
            
            mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
            mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
            mlflow.sklearn.log_model(model, "elnet_model")

            fs.log_model(
                model=model,
                artifact_path="el_wine_quality_prediction",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name="elnet_wine_model",
            )
 
train_model(Xtrain, Xtest, Ytrain, Ytest, training_set, fs)


# COMMAND ----------

batch_input_df = target_df.drop("quality") # Drop the label column
 
predictions_df = fs.score_batch("models:/elnet_wine_model/staging", batch_input_df)
type(predictions_df)

from pyspark.sql.functions import isnan
model_uri = 'dbfs:/databricks/mlflow-tracking/3311970642464864/46b7107d2bff458691ab1cef35a67b88/artifacts/el_wine_quality_prediction'
#predictions_df.where(isnan(col("prediction")))
predictions_df.select("prediction").show()
#pandasDF = predictions_df.toPandas()
#predictions_df = predictions_df.reset_index()
type(predictions_df["wine_id", "prediction"])
#display(batch_input_df)

# COMMAND ----------



# COMMAND ----------

# load input data table as a Spark DataFrame
#input_data = spark.table(batch_input_df)
model_udf = mlflow.pyfunc.spark_udf(spark ,model_uri=model_uri)
#model_uri='models:/elastic_wine_model/staging'
df = batch_input_df.withColumn("prediction", model_udf(())
                              )

# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

score_df = spark.table("wine_data")
preds = (score_df.withColumn("Qualityprediction", pyfunc_udf()))

# COMMAND ----------

import mlflow
import mlflow.spark
import mlflow.sklearn
import mlflow.azureml
run_id = '46b7107d2bff458691ab1cef35a67b88'
#model_uri = 'runs:/'+run_id+'/model' 
model_uri = 'dbfs:/databricks/mlflow-tracking/3311970642464864/46b7107d2bff458691ab1cef35a67b88/artifacts/el_wine_quality_prediction'
experiment = 'wineElasticnet'
azure_service, azure_model = mlflow.azureml.deploy(model_uri=model_uri, service_name = experiment+"_service", 
                                                   workspace="tests",
                                                  synchronous = True)
#azure_service, azure_model = mlflow.deployments.run_local('target', model_uri=model_uri, name = experiment+"_service")

# COMMAND ----------


#mlflow.deployments.run_local()

# COMMAND ----------

from mlflow.deployments import get_deploy_client
deployment_client = get_deploy_client(mlflow.get_tracking_uri())

# COMMAND ----------

import mlflow
logged_model = 'runs:/06ff4f3e17994056bad20b574425b19f/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(training_set))

# COMMAND ----------

