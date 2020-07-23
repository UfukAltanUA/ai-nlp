# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spam').getOrCreate()



structureNLP= [['Tokenizer'],
               ['RE (If needed)'],
               ['Stopwords'],
               ['NGram (If needed)'],
               ['HashingTF'],
               ['IDF'],
               ['CountVectorizer']]        



df = spark.read.csv('/FileStore/tables/SMSSpamCollection',inferSchema=True,sep='\t')
df.show(truncate=False)



df = df.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
df.show()



from pyspark.sql.functions import length
df = df.withColumn('length', length(df['text']))
df.show()



df.groupby('class').mean().show()



structureNLP



from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer



tokenizer = Tokenizer(inputCol='text', outputCol='tokens')
remover = StopWordsRemover(inputCol='tokens', outputCol='stopTokens')
cv = CountVectorizer(inputCol='stopTokens', outputCol='vectors')
idf = IDF(inputCol='vectors', outputCol='tfidf')
indexer = StringIndexer(inputCol='class', outputCol='label')



from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['tfidf','length'], outputCol='features')



from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
classifier = NaiveBayes()
pipeline = Pipeline(stages=[indexer,tokenizer,remover,cv, idf, assembler])



cleaner = pipeline.fit(df)
final_data = cleaner.transform(df)
final_data = final_data.select('label', 'features')
final_data.show()



train,test = final_data.randomSplit([0.7,0.3])



spamDetector = classifier.fit(train)
results = spamDetector.transform(test)
results.show()



from pyspark.ml.evaluation import MulticlassClassificationEvaluator
multiEval = MulticlassClassificationEvaluator()
acc = multiEval.evaluate(results)



print('Accuracy')
print(acc)




