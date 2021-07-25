import pandas as pd
import os
from logistic_regression.model import LR
from rating_space_comparison.model import RatingSpaceComparison
from bayesian_tree.model import BN
import pickle
from anova import Anova





rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(os.path.join('ml-1m', 'ratings.dat'), sep='::', header=None, names=rnames, engine='python')
rating_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0).astype(int)
users = pd.read_pickle(os.path.join('ml-1m', 'users.pkl'))
users.set_index(users.user_id, inplace=True)

anova = Anova(rating_matrix, users)
movies = anova.movies

print(movies)
# Logistic Regression
lr = LR(rating_matrix, users)
lr.movies = movies
lr.filter_movie()
data_output = open('lr_result.pkl','wb')
pickle.dump(lr.reports,data_output)
data_output.close()

# Rating space Comparison
# rs = RatingSpaceComparison(rating_matrix, users)
# rs.movies = movies
# rs.filter_movie()
# rs_result = rs.result
# print(rs_result)
# data_output = open('rs_result.pkl','wb')
# pickle.dump(rs_result,data_output)
# data_output.close()

# BN
bn = BN(rating_matrix, users, 'chow-liu')
bn.filter_movie()
reports = bn.reports
data_output = open('bn_result.pkl','wb')
pickle.dump(reports,data_output)
data_output.close()
