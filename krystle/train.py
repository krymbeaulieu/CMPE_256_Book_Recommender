import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import time

try:
    from surprise import NormalPredictor, KNNBasic, NMF, SlopeOne, SVD, Dataset, Reader
    from surprise.model_selection import cross_validate
except ModuleNotFoundError:
  raise ModuleNotFoundError("cannot find surprise, please pip install surprise")

try:
    import kagglehub
except ModuleNotFoundError:
    raise ModuleNotFoundError("cannot find kagglehub, please pip install kagglehub")

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments with defaults.")

    parser.add_argument("--ds", type=str, default="arashnic/book-recommendation-dataset",
                        help="kaggle Dataset path or identifier (default: arashnic/book-recommendation-dataset).")
    parser.add_argument("--rating_scale", type=int, nargs=2, default=(1, 10),
                        help="Rating scale as a tuple (default: (1, 10)).")
    parser.add_argument("--use_explicit", type=bool, default=True,
                        help="Use explicit ratings (default: True). aka what is in rating scale")
    parser.add_argument("--cv", type=int, default=3,
                        help="Number of cross-validation folds (default: 3).")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Verbose output (default: True).")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of jobs to run in parallel (default: -1). -1 is all cpu, 1 is single threaded")
    parser.add_argument("--k", type=int, default=10,
                          help="k neighbors. surprise knn default is 40 but may hit memory issues so default is 10 now. (default: 10).")

    return parser.parse_args()
  
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
                               n_jobs=-1,k=10):
  
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
  sim_options = {'name': 'cosine', 'user_based': True}
  algo = KNNBasic(k=k,verbose=True,sim_options=sim_options)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['User-based Collaborative Filtering'] = scores
  
  print("KNNBasic: item-based collab filter")
  sim_options = {'name': 'cosine', 'user_based': False}
  algo = KNNBasic(k=k,verbose=True,sim_options=sim_options)
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
  #example: python3 train.py --ds arashnic/book-recommendation-dataset --rating_scale 1 10 --use_explicit --cv 3 --verbose True --n_jobs -1 --k 10
  args = parse_args()
  do_collaborative_filtering(args.ds,args.rating_scale,args.use_explicit,args.cv,args.verbose,args.n_jobs,args.k)
