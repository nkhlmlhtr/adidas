#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pyspark


# In[40]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, count, to_timestamp, year, regexp_extract, coalesce, from_json, col, when, explode, length, size, lit, current_date, trim, dense_rank, regexp_replace, split, countDistinct, to_date, concat_ws
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window


# In[41]:


spark = SparkSession.builder.appName('Adidas_CaseStudy').master('local[*]').getOrCreate()


# In[42]:


spark


# Reading Dataset

# In[43]:


base_df = spark.read.json('/tmp/ol_cdump.json')


# Number of rows in raw data set

# In[44]:


print('Number of rows in raw data set : '+ str(base_df.count()))


# Schema

# In[45]:


base_df.printSchema()


# # Data profiling

# In[46]:


## Merging Similar columns including authors which will get used further.
base_df = base_df.withColumn('dewey_number',coalesce('dewey_decimal_class','dewey_number')).withColumn('oclc_number',coalesce('oclc_number','oclc_numbers')).withColumn('subject_time',coalesce('subject_time','subject_times')).withColumn('work_title',coalesce('work_title','work_titles')).withColumn('publish_date',coalesce('publish_date','first_publish_date')).withColumn('authors', coalesce('authors.key','authors.author.key')).drop('dewey_decimal_class','oclc_numbers','subject_times','work_titles','first_publish_date')


# In[47]:


### Trimming Title to remove extra spaces
### Extracting date value for last_modified_datetime
### Extracting publish year which will be used to filter.
base_df = base_df.withColumn('title',trim('title')).withColumn('last_modified_datetime',to_timestamp(base_df['last_modified.value'])).drop('last_modified').withColumn('publish_year',regexp_extract('publish_date', r'(\d{4})', 1))
#withColumn('authors', regexp_replace('authors', '/authors/', '')).\
#withColumn('authors',split(col('authors'),','))


# In[48]:


schema = StructType([StructField("type", StringType(), True),
                    StructField("value", StringType(), True)]
                    )


# In[49]:


##Extracting values from key, value data strings.
base_df = base_df.withColumn('description',when(col('description').startswith('{"type":'),from_json(base_df['description'],schema)['value']).otherwise(col('description'))).withColumn('first_sentence',when(col('first_sentence').startswith('{"type":'),from_json(base_df['first_sentence'],schema)['value']).otherwise(col('first_sentence'))).withColumn('notes',when(col('notes').startswith('{"type":'),from_json(base_df['notes'],schema)['value']).otherwise(col('notes'))).withColumn('bio',when(col('bio').startswith('{"type":'),from_json(base_df['bio'],schema)['value']).otherwise(col('bio')))


# In[50]:


###All these fields mostly have Null values (Not Required). 
base_df = base_df.drop('full_title','fuller_name','ia_box_id', 'ia_loaded_id', 'isbn_invalid', 'isbn_odd_length', 'website')


# In[51]:


###Filtering data publish year from 1950 to current year.
base_df = base_df.filter((col('publish_year') <= lit(year(current_date()))) & (col('publish_year') >= lit('1950')))


# In[52]:


print('Number of rows in filtered data set : '+ str(base_df.count()))


# ## Book with Most Pages

# In[53]:


w = Window.orderBy(col('number_of_pages').desc())
first_df = base_df.select('title','number_of_pages',dense_rank().over(w).alias('page_rank'))


# In[54]:


first_df.filter(col('page_rank') == 1).select('title','number_of_pages').show(9,False)


# # Top 5 genres

# In[55]:


second_df = base_df.select('title',explode('genres').alias('genres'))


# In[56]:


second_df = second_df.select('title',trim(regexp_replace('genres', '[^A-Za-z0-9, ]','')).alias('genres'))


# In[57]:


second_df.groupBy('genres').agg(count('title').alias('book_count')).orderBy(col('book_count').desc()).show(5,False)


# # Top 5 authors who (co-)authored  most books

# In[58]:


third_df = base_df.select('title','authors')


# In[59]:


##This df will have list of all the authors who have co-authored
third_df = third_df.filter(size('authors') > 1)


# In[60]:


third_df = third_df.select('title',explode('authors').alias('authors'))


# In[61]:


third_df.groupBy('authors').agg(count('title').alias('book_count')).orderBy(col('book_count').desc()).show(5,False)


# # Number of authors that published at least one book. (By Publish Year)

# In[62]:


fourth_df = base_df.select('publish_year',explode('authors').alias('authors')).distinct()


# In[63]:


fourth_df.groupBy('publish_year').agg(count('authors').alias('no_of_authors')).orderBy(col('publish_year').desc()).show(9999,False)


# # Number of authors and number of books published per month for years between 1950 and 1970

# In[64]:


fifth_df = base_df.filter((col('publish_year') >= lit('1950')) & (col('publish_year') <= lit('1970'))).select(explode('authors').alias('authors'),'title','publish_year','publish_date')


# In[65]:


fifth_df = fifth_df.withColumn('publish_month', regexp_extract('publish_date',r'([A-Z][a-z]+)', 1))


# In[66]:


fifth_df = fifth_df.filter(~(col('publish_month') == lit('')))


# In[67]:


fifth_df.groupBy(concat_ws(' ','publish_month','publish_year').alias('month_year')).agg(countDistinct('authors').alias('authors'),countDistinct('title').alias('books')).orderBy(concat_ws(' ','publish_month','publish_year')).show(99,False)


# In[ ]:


#fifth_df.filter((col('publish_year') == lit('1965')) & (col('publish_month') == lit('June'))).show(4,False)


# In[ ]:




