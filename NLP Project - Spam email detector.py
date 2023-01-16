#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('NLP').getOrCreate()

data2 = spark.read.csv('/users/f/desktop/smsspamdata', inferSchema = True, sep = '\t') #As our data is separated by tabs we should read the data with the separation '\t'.

data2.show()


# In[ ]:





# In[5]:


data2 = data2.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1','text')#Rename the headers 

data2.show()

from pyspark.sql.functions import length 

data2 = data2.withColumn('length',length(data2['text']))#We need the lenghth of the data 

data2.show()

data2.groupBy('class').mean().show()#separated by groups (average)

from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer)

tokenizer = Tokenizer(inputCol = "text", outputCol = "token_text")

stop_remove = StopWordsRemover(inputCol = 'token_text', outputCol="stop_tokens")

count_vec = CountVectorizer(inputCol = 'stop_tokens', outputCol = 'c_vec')

idf = IDF(inputCol = 'c_vec', outputCol = 'tearmf_inversf')

ham_spam_to_numbers = StringIndexer(inputCol = 'class', outputCol = "label")#converts strings to numbers to make the spark understand 

from pyspark.ml.feature import VectorAssembler 

clean_up_data = VectorAssembler(inputCols=['tearmf_inversf', 'length'],outputCol = 'features')

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()

from pyspark.ml import Pipeline 

data_prep_pipeline = Pipeline(stages = [ham_spam_to_numbers, tokenizer, stop_remove, count_vec, idf, clean_up_data])

cleaner = data_prep_pipeline.fit(data2)

clean_data2 = cleaner.transform(data2)

clean_data2 = clean_data2.select('label','features')

clean_data2.show()

training_data2, test_data2 = clean_data2.randomSplit([0.7, 0.3])

spam_detection2 = nb.fit(training_data2)


# In[6]:


spam_detection2 = nb.fit(training_data2)


# In[7]:


test_results = spam_detection2.transform(test_data2)


# In[8]:


test_results.show()


# In[11]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[13]:


acc_eval2 = MulticlassClassificationEvaluator()
acc2 = acc_eval2.evaluate(test_results)
print("Accuracy of model at predicting spam is: {}".format(acc2))


# In[15]:


datax = [('Nah I dont think he goes to usf, he lives around here thoug','NULL')]


# In[16]:


datax


# In[17]:


columns = ["text","length"]


# In[18]:


dfx = spark.createDataFrame(data=datax, schema = columns)


# In[19]:


dfx.show()


# In[20]:


datax = dfx.withColumn('length',length(dfx['text']))#We need the lenghth of the data


# In[21]:


datax.show()


# In[22]:


tokenizer2 = Tokenizer(inputCol = "text", outputCol = "token_text2")

stop_remove2 = StopWordsRemover(inputCol = 'token_text2', outputCol="stop_tokens2")

count_vec2 = CountVectorizer(inputCol = 'stop_tokens2', outputCol = 'c_vec2')

idf2 = IDF(inputCol = 'c_vec2', outputCol = 'tf_idf')


# In[23]:


clean_up_datax = VectorAssembler(inputCols=['tf_idf','length'],outputCol = 'features')


# In[24]:


data_prep_pipex = Pipeline(stages=[tokenizer2,stop_remove2,count_vec2,idf2,clean_up_datax])


# In[25]:


cleanx = data_prep_pipex.fit(datax)


# In[26]:


cleandatax = cleanx.transform(datax)


# In[27]:


cleandatax.show()


# In[28]:


cleandatax2 = cleandatax.select('features')


# In[29]:


predictionx = spam_detection2.transform(cleandatax)


# In[30]:


predictionx


# In[ ]:


predictionx.show()

