try:
    import kagglehub
except ModuleNotFoundError:
    raise ModuleNotFoundError("cannot find kagglehub, please pip install kagglehub")
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import time
try:
    from surprise import NormalPredictor, KNNBasic, NMF, SlopeOne, SVD, Dataset, Reader
    from surprise.model_selection import cross_validate
except ModuleNotFoundError:
  raise ModuleNotFoundError("cannot find surprise, please pip install surprise")

# Download latest version

def download_dataset(ds = "arashnic/book-recommendation-dataset"):
  path = kagglehub.dataset_download(ds)
  print("Path to dataset files:", path)
  return path

def setup_dataset(ds):
  # download dataset & open it up
  if ds == "arashnic/book-recommendation-dataset":
    path = download_dataset(ds)
    
    user_path = list(Path(path).glob("*User*.csv"))[0]
    users_df = pd.read_csv(user_path)
  
    rating_path = list(Path(path).glob("*Rating*.csv"))[0]
    ratings_df = pd.read_csv(rating_path)
    
    
    print("merging dataframes")
    user_rating_df = pd.merge(users_df[["User-ID",]],ratings_df,on="User-ID")
    print(user_rating_df.head())
    return user_rating_df
  else:
    raise NotImplementedError(f"dataset handling for {ds} is not implemented yet.")
  
def do_collaborative_filtering(ds = "arashnic/book-recommendation-dataset", 
                               rating_scale = (1, 10),
                               use_explicit=True,
                               cv=3,
                               verbose=True,
                               n_jobs=-1):
  
  # download dataset, load, merge data
  user_rating_df = setup_dataset(ds)
  print(f"using explicit rating {rating_scale}")
  

  print("collaborative filtering")
  results = {}
  errordict = {'RMSE':{'key':'test_rmse','name':'Root Mean Square Error'},
               'MAE':{'key':'test_mae','name':'Mean Absolute Error'}}
 
  # reader for surprise module
  reader = Reader(rating_scale=rating_scale)
  # figure out what dataset to use
  if ds == "arashnic/book-recommendation-dataset":
    if use_explicit:
      print("using explicit dataset")
      explicit_df = user_rating_df[user_rating_df['Book-Rating'] > 0]
      print("explicit min,max: ",explicit_df['Book-Rating'].min(),", ",explicit_df['Book-Rating'].max())
      data = Dataset.load_from_df(explicit_df[['ISBN','User-ID','Book-Rating']],reader)
      # clear up some memory
      del user_rating_df
    else:
      print("using implicit dataset")
      implicit_df = user_rating_df[user_rating_df['Book-Rating'] == 0]
      print("implicit min,max: ",implicit_df['Book-Rating'].min(),", ",implicit_df['Book-Rating'].max())
      data = Dataset.load_from_df(implicit_df[['ISBN','User-ID','Book-Rating']],reader)
      # clear up some memory
      del user_rating_df
  else:
    raise NotImplementedError(f"dataset {ds} not implemented yet for collaborative filtering.")

 
  
  print("normal predictor")
  algo = NormalPredictor()
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Random_baseline'] = scores
  
  print("KNNBasic: user-based collab filter")
  algo = KNNBasic(verbose=True)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['User-based Collaborative Filtering'] = scores
  
  print("KNNBasic: item-based collab filter")
  algo = KNNBasic(verbose=True,sim_options={'user_based':False})
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Item-based Collaborative Filtering'] = scores
  
  print("NMF")
  algo = NMF(random_state=0)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Non-negative Matrix Factorization'] = scores
  
  print("SlopeOne")
  algo = SlopeOne()
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['SlopeOne Collaborative Filtering'] = scores

if __name__ == "__main__":
  do_collaborative_filtering()
