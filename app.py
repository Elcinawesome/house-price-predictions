import pyspark
from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('housing_price_model').getOrCreate()
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)
house=spark.read.csv('data.csv',inferSchema=True,header=True)
# features=spark.read.csv('features.csv',inferSchema=True,header=True)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
    inputCols=['sqft','bhk','bath'],
        outputCol="features")
    
output = assembler.transform(house)
final_data = output.select("features",'price')
train_data,test_data = final_data.randomSplit([0.7,0.3])
from pyspark.ml.regression import LinearRegression
# Create a Linear Regression Model object
lr = LinearRegression(labelCol='price')
# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data)
#
#a=float(3.6)
#b=1
#c=1
#
#hs_gpa = sqlContext.createDataFrame([(1,Vectors.dense(a,b,c)),],['index','features'])
#
#predictions_test=lrModel.transform(hs_gpa)
#
#output_pred=predictions_test.collect()


from flask import Flask, render_template, request
import pickle
import numpy as np
#filename = 'msft_model.pkl'
#classifier = pickle.load(open(filename, 'rb'))







app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sqft = int(request.form['sqft'])
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])
        
        
        
        
        hs_gpa = sqlContext.createDataFrame([(1,Vectors.dense(sqft,bhk,bath)),],['index','features'])
        
        predictions_test=lrModel.transform(hs_gpa)
        
        output_pred=predictions_test.collect()
    
        
#        data = np.array([[sqft, bhk, bath]])
#        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=output_pred)

if __name__ == '__main__':
	app.run(debug=True)
