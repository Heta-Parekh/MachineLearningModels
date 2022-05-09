# **Ridge Regression Model in PySpark**
## Download the csv file from https://www.kaggle.com/ANANAYMITAL/US-USED-CARS-DATASET and upload to the hadoop oracle server using scp file_name username@ip.address
## Import Spark SQL and Spark ML Libraries
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType,IntegerType
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

"""##Create the Spark Session to load the dataset"""

PYSPARK_CLI = True # conditional statement to run only at shell
if PYSPARK_CLI:
 sc = SparkContext.getOrCreate()
 spark = SparkSession(sc)

"""## Load Source Data"""

#File location and type
file_location = "/tmp/used_cars_data.csv"
#file_location = "/Users/kangjoin/Downloads/used_cars_data 2.csv"
file_type = "csv"
 
#CSV options
infer_schema = "TRUE"
first_row_is_header = "TRUE"
delimiter = ","
 
#The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
 
df.show()

"""## Read csv file from HDFS of Oracle Big Data cluster"""
sc2 = SparkSession.builder.master('local[*]').appName('used_cars_data').getOrCreate()
file_df = sc2.read.csv(file_location,header=True)
file_df.show()

"""## Prepare the data"""

df_new = file_df.select(col('engine_displacement'),col('frame_damaged') ,col('has_accidents') ,col('horsepower'),col('isCab'),col('is_new'),col('mileage'),col('power'),col('price'),col('seller_rating'),col('sp_id'),col('make_name'),col('daysonmarket'))
df_new = df_new.withColumn("engine_displacement",col("engine_displacement").cast(DoubleType()))
df_new = df_new.withColumn("horsepower",col("horsepower").cast(DoubleType()))
df_new = df_new.withColumn("power",col("power").cast(DoubleType()))
df_new = df_new.withColumn("mileage",col("mileage").cast(IntegerType()))
df_new = df_new.withColumn("price",col("price").cast(IntegerType()))
df_new = df_new.withColumn("seller_rating",col("seller_rating").cast(DoubleType()))

cols = ['is_new']
col2 = ['frame_damaged','has_accidents','isCab']

df_new= reduce(lambda df_new, c: df_new.withColumn(c, F.when(df_new[c] == 'False', 0).otherwise(1)), cols, df_new)
df_new=  df_new.na.fill(value=0,subset=["mileage"])

df_new = reduce(lambda df_new, c: df_new.withColumn(c, F.when(df_new[c]== 'False', 2).when(df_new[c]== 'True', 0).otherwise(1)), col2, df_new)
df_new= df_new.na.fill(value=0,subset=["engine_displacement"])
df_new= df_new.na.fill(value=0,subset=["horsepower"])
df_new= df_new.na.fill(value=0,subset=["power"])
df_new= df_new.na.fill(value=0,subset=["seller_rating"])
df_new= df_new.na.fill(value=0,subset=["price"])

df_new = df_new.withColumn("is_new",col("is_new").cast(IntegerType()))
df_new = df_new.withColumn("frame_damaged",col("frame_damaged").cast(IntegerType()))
df_new = df_new.withColumn("has_accidents",col("has_accidents").cast(IntegerType()))
df_new = df_new.withColumn("isCab",col("isCab").cast(IntegerType()))
df_new = df_new.select('*').where(col("price")>0)
df_new = df_new.select('*').where(col("price")<10000000)
df_new = df_new.select('*').where(col("engine_displacement")>0)
df_new = df_new.select('*').where(col("horsepower")>0)

df_new = df_new.drop("power")
df_new.printSchema()
df_new.show()
"""## Split the Data"""

splits = df_new.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, "Testing Rows:", test_rows)

"""## Define the Pipeline"""
assembler = VectorAssembler(inputCols =["engine_displacement","is_new", "mileage", "frame_damaged", "has_accidents", "seller_rating","isCab","horsepower"], outputCol="features")
#assembler = VectorAssembler(inputCols =["engine_displacement","is_new", "mileage", "frame_damaged", "has_accidents", "seller_rating","power","isCab","horsepower"], outputCol="features")

"""## Train a Regression Model"""

Rlr = LinearRegression(labelCol="price",featuresCol="features",maxIter=10, regParam=0)
pipeline = Pipeline(stages=[assembler, Rlr])
model = pipeline.fit(train)


"""## Test the Model"""


prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "price")
predicted = predicted.drop("features")
predicted.show()

"""## Calculate the RMSE and R2


"""

rlr_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="price",metricName="r2")

print("R Squared (R2) on test data = %g" %rlr_evaluator.evaluate(prediction))

rlr_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % rlr_evaluator.evaluate(prediction))
