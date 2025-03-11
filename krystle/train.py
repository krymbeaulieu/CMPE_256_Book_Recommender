import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import time
import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

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
    parser.add_argument("--allow_install", type=bool, default=False,
                        help="allow pip install (default: False).")

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

def run_normal(results, errordict, data, cv=3, verbose=True,n_jobs=-1):
  print("\nnormal predictor")
  algo = NormalPredictor()
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Random_baseline'] = scores
  return results

def run_knn_user(results, errordict, data, cv=3, verbose=True, n_jobs=-1, k=10):
  print(f"\nKNNBasic: user-based collab filter predictor, k = {k}")
  sim_options = {'name': 'cosine', 'user_based': True}
  algo = KNNBasic(k=k,verbose=verbose,sim_options=sim_options)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['User-based Collaborative Filtering'] = scores
  return results

def run_knn_items(results, errordict, data, cv=3, verbose=True, n_jobs=-1, k=10):
  print(f"\nKNNBasic: item-based collab filter predictor, k = {k}")
  sim_options = {'name': 'cosine', 'user_based': False}
  algo = KNNBasic(k=k,verbose=True,sim_options=sim_options)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Item-based Collaborative Filtering'] = scores
  return results

def run_nmf(results, errordict, data, cv=3, verbose=True, n_jobs=-1):
  print("\nNMF predictor")
  algo = NMF(random_state=0)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Non-negative Matrix Factorization'] = scores
  return results

def run_slopeOne(results, errordict, data, cv=3, verbose=True, n_jobs=-1):
  print("\nSlopeOne predictor")
  algo = SlopeOne()
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['SlopeOne Collaborative Filtering'] = scores
  return results
  
def run_svd(results, errordict, data, cv=3, verbose=True, n_jobs=-1):
  algo = SVD(random_state=42)
  scores = cross_validate(algo, data, measures=list(errordict.keys()), cv=cv, verbose=verbose,n_jobs=n_jobs)
  results['Funk Matrix Factorization'] = scores
  return results
  
def gen_fig_results(results,errordict):

  for errortype in list(errordict.keys(),cv):
    tags = []
    scrs = []
    algo = []
    
    for k,(key,value) in enumerate(results.items()):
        if key=='Random_baseline': continue
        tags.append('Algorithm '+str(k))
        algo.append(key)
        scrs.append(value[errordict[errortype]['key']].mean())
    bar_colors = ['tab:red','tab:blue','tab:purple','tab:green','tab:orange']*5
    
    fig, ax = plt.subplots()
    ax.bar(tags, scrs, label=algo, color=bar_colors[:len(tags)])
    ax.legend()
    fig.set_size_inches(10,4)
    minval = [x[errordict[errortype]['key']].mean() for x in results.values()][1:]
    ax.set_ylim(int(min(minval)*10)/10.0,int(max(minval)*10+1)/10.0)
    ax.set_ylabel(errordict[errortype]['name'])
    ax.legend(title='Recommendation Algorithm')
    fn = f"results_{errordict[errortype]['key']}_{cv}.png"
    plt.savefig(fn)
    print(f"results saved to {fn}")

def setup_data_surprise(user_rating_df,rating_scale,ds,use_explicit):
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
  return data, reader

def do_collaborative_filtering(ds = "arashnic/book-recommendation-dataset", 
                               rating_scale = (1, 10),
                               use_explicit=True,
                               cv=3,
                               verbose=True,
                               n_jobs=-1,k=10):
  
  # download dataset, load, merge data
  user_rating_df = setup_dataset(ds)
  print(f"using rating {rating_scale}")
  data,reader = setup_data_surprise(user_rating_df,rating_scale,ds,use_explicit)

  print("collaborative filtering")
  results = {}
  errordict = {'RMSE':{'key':'test_rmse','name':'Root Mean Square Error'},
               'MAE':{'key':'test_mae','name':'Mean Absolute Error'}}
 
 

 
  results = run_normal(results, errordict, data, cv, verbose,n_jobs)

  # results = run_knn_user(results, errordict, data, cv, verbose, n_jobs, k)

  # results = run_knn_item(results, errordict, data, cv, verbose, n_jobs, k)

  results = run_nmf(results, errordict, data, cv, verbose,n_jobs)

  results = run_slopeOne(results, errordict, data, cv, verbose,n_jobs)

  results = run_svd(results, errordict, data, cv, verbose,n_jobs)
  
  gen_fig_results(results, errordict, cv)

def find_best(ds = "arashnic/book-recommendation-dataset", 
                               rating_scale = (1, 10),
                               use_explicit=True,
                               cv=3,
                               verbose=True,
                               n_jobs=-1):
  
  # download dataset, load, merge data
  user_rating_df = setup_dataset(ds)
  print(f"using rating {rating_scale}")
  data,reader = setup_data_surprise(user_rating_df,rating_scale,ds,use_explicit)

  print("grid search for SVD")
  results = {}
  errordict = {'RMSE':{'key':'test_rmse','name':'Root Mean Square Error'},
               'MAE':{'key':'test_mae','name':'Mean Absolute Error'}}
  
  param_grid = {
    
    'n_factors': [70, 80, 90, 100, 110, 120, 130, 140, 150, 160], 
    'n_epochs': [20,100], 
    'reg_all': [.02,0.1],
    'biased': [True,False],
    'lr_all': [1e-3,.005]
  }
  print(f"grid search with {param_grid} for SVD")
  gs = GridSearchCV(SVD, param_grid, measures=list(errordict.keys()), cv=cv,n_jobs=n_jobs)
  gs.fit(data)
  # best score
  print(f"best score: {gs.best_score}")
  
  # combination of parameters that gave the best RMSE score
  print(f'best params RMSE: {gs.best_params["rmse"]}')
  print(f'best params MAE: {gs.best_params["mae"]}')

if __name__ == "__main__":
  #example: python3 train.py --ds arashnic/book-recommendation-dataset --rating_scale 1 10 --use_explicit --cv 3 --verbose True --n_jobs -1 --k 10 --allow_install False
  args = parse_args()
  try:
      from surprise import NormalPredictor, KNNBasic, NMF, SlopeOne, SVD, Dataset, Reader
      from surprise.model_selection import cross_validate
      from surprise.model_selection.search import GridSearchCV
  except ModuleNotFoundError:
    if args.allow_install:
      install('surprise')
      from surprise import NormalPredictor, KNNBasic, NMF, SlopeOne, SVD, Dataset, Reader
      from surprise.model_selection import cross_validate
      from surprise.model_selection.search import GridSearchCV
    else:
      raise ModuleNotFoundError(f"cannot find surprise, please pip install surprise or turn on allow install and rerun (currently: {allow_install})")
  
  try:
      import kagglehub
  except ModuleNotFoundError:
      if args.allow_install:
        install('kagglehub')
        import kagglehub
      else:
        raise ModuleNotFoundError(f"cannot find kagglehub, please pip install kagglehub or turn on allow install and rerun (currently: {allow_install})")

  print("find best")
  find_best(args.ds,args.rating_scale,args.use_explicit,args.cv,args.verbose,args.n_jobs)      
  
  # print("do collaborative filtering")
  # do_collaborative_filtering(args.ds,args.rating_scale,args.use_explicit,args.cv,args.verbose,args.n_jobs,args.k)
