#!/usr/bin/env python
# coding: utf-8

# ## Homework
# 
# In this homework, we'll deploy the ride duration model in batch mode. Like in homework 1, we'll use the Yellow Taxi Trip Records dataset. 

# ## Q1. Notebook
# 
# We'll start with the same notebook we ended up with in homework 1.
# We cleaned it a little bit and kept only the scoring part.
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# * 1.24
# * **6.24**
# * 12.28
# * 18.28

# In[2]:


import pickle
import pandas as pd
import numpy as np

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()


    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

std = np.std(y_pred)
print(f"The standard deviation of the predicted duration: {std}")


# ## Q2. Preparing the output
# 
# Like in the course videos, we want to prepare the dataframe with the output. 
# 
# First, let's create an artificial `ride_id` column:
# 
# ```python
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# ```
# 
# Next, write the ride id and the predictions to a dataframe with results. 
# 
# Save it as parquet:
# 
# ```python
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# 
# What's the size of the output file?
# 
# * 36M
# * 46M
# * 56M
# * **66M**
# 
# __Note:__ Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the
# dtypes of the columns and use `pyarrow`, not `fastparquet`. 
# 

# In[4]:


import os

year = 2023
month = 3

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Create a new dataframe with ride_id and predicted duration
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

output_file = 'output.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

file_size = os.path.getsize(output_file)
print(f"The size of the Parquet file is: {file_size / (1024 * 1024):.2f} MB")


# ## Q3. Creating the scoring script
# 
# Now let's turn the notebook into a script. 
# 
# Which command you need to execute for that?

# In bash:
# 
# > jupyter nbconvert --to script hw_4.ipynb

# ## Q4. Virtual environment
# 
# Now let's put everything into a virtual environment. We'll use pipenv for that.
# 
# Install all the required libraries. Pay attention to the Scikit-Learn version: it should be the same as in the starter
# notebook.
# 
# After installing the libraries, pipenv creates two files: `Pipfile`
# and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
# dependencies we use for the virtual env.
# 
# What's the first hash for the Scikit-Learn dependency?

# 1. Install pipenv if you don’t have it already (in bash):
# 
# > pip install pipenv
# 
# 2. Create the virtual environment and install scikit-learn using the correct version (e.g., 1.2.2, as used in the starter notebook) (in bash):
# 
# > pipenv install scikit-learn==1.2.2
# 
# 3. After the installation, pipenv will generate two files in your directory: Pipfile and Pipfile.lock
# 
# 4. Open Pipfile.lock in a text editor. It's a JSON file.
# 
# 5. Search for the "scikit-learn" section. It will look something like this (in the .json):
#  
#  ```
# "scikit-learn": {
#     "version": "==1.2.2",
#     "hashes": [
#         "sha256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
#         ...
#     ]
# }
# ```
# 6. Copy the first hash listed in the "hashes" array.

# ## Q5. Parametrize the script
# 
# Let's now make the script configurable via CLI. We'll create two 
# parameters: year and month.
# 
# Run the script for April 2023. 
# 
# What's the mean predicted duration? 
# 
# * 7.29
# * 14.29
# * 21.29
# * 28.29
# 
# Hint: just add a print statement to your script.

# To run for April 2023 (in bash):
# 
# > python H4_Q5.py --year 2023 --month 4
# 

# ## Q6. Docker container 
# 
# Finally, we'll package the script in the docker container. 
# For that, you'll need to use a base image that we prepared. 
# 
# This is what the content of this image is:
# 
# ```dockerfile
# FROM python:3.10.13-slim
# 
# WORKDIR /app
# COPY [ "model2.bin", "model.bin" ]
# ```
# 
# Note: you don't need to run it. We have already done it.
# 
# It is pushed to [`agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim`](https://hub.docker.com/layers/agrigorev/zoomcamp-model/mlops-2024-3.10.13-slim/images/sha256-f54535b73a8c3ef91967d5588de57d4e251b22addcbbfb6e71304a91c1c7027f?context=repo),
# which you need to use as your base image.
# 
# That is, your Dockerfile should start with:
# 
# ```dockerfile
# FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
# 
# # do stuff here
# ```
# 
# This image already has a pickle file with a dictionary vectorizer
# and a model. You will need to use them.
# 
# Important: don't copy the model to the docker image. You will need
# to use the pickle file already in the image. 
# 
# Now run the script with docker. What's the mean predicted duration
# for May 2023? 
# 
# * 0.19
# * 7.24
# * 14.24
# * 21.19

# In[ ]:





# ## Bonus: upload the result to the cloud (Not graded)
# 
# Just printing the mean duration inside the docker image 
# doesn't seem very practical. Typically, after creating the output 
# file, we upload it to the cloud storage.
# 
# Modify your code to upload the parquet file to S3/GCS/etc.
# 

# In[ ]:





# ## Bonus: Use an orchestrator for batch inference
# 
# Here we didn't use any orchestration. In practice we usually do.
# 
# * Split the code into logical code blocks
# * Use a workflow orchestrator for the code execution

# In[ ]:





# ## Publishing the image to dockerhub
# 
# This is how we published the image to Docker hub:
# 
# ```bash
# docker build -t mlops-zoomcamp-model:2024-3.10.13-slim .
# docker tag mlops-zoomcamp-model:2024-3.10.13-slim agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
# 
# docker login --username USERNAME
# docker push agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
# ```
# 
# This is just for your reference, you don't need to do it.

# In[ ]:




