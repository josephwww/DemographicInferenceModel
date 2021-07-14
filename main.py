import pandas as pd
import os
from logistic_regression.model import LR
from rating_space_comparison.model import RatingSpaceComparison
from bayesian_tree.model import BN
import pickle

def remove_duplicate_movies():
    temp = []
    for movie in movies:
        if movie not in temp:
            temp.append(movie)
    return temp



rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(os.path.join('ml-1m', 'ratings.dat'), sep='::', header=None, names=rnames, engine='python')
rating_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0).astype(int)
users = pd.read_pickle(os.path.join('ml-1m', 'users.pkl'))
users.set_index(users.user_id, inplace=True)

age_movie = [3033, 1968, 919, 2005, 2918, 1196, 1544, 1240, 1517, 2571, 969, 2369, 1213, 2959, 2710, 2413, 104,
                     2371, 34, 2858]
gender_movie = [1088, 3421, 1201, 1380, 1221, 2791, 2028, 1265, 1028, 3552, 589, 920, 17, 858, 1200, 1196, 2657,
                        1240, 110, 1214]
occu_movie = [3578, 3033, 1240, 2571, 110, 1214, 2028, 2762, 1207, 2683, 1097, 597, 589, 1210, 1544, 1732, 3753,
                      919, 736, 3421]
movies = [movie for movies in list(zip(age_movie, gender_movie, occu_movie)) for movie in movies]
movies = remove_duplicate_movies()

# Logistic Regression
# lr = LR(rating_matrix, users)
# lr.movies = movies
# lr.filter_movie()
# lr_result = lr.result
# print(lr_result)
# data_output = open('lr_result.pkl','wb')
# pickle.dump(lr_result,data_output)
# data_output.close()

# Rating space Comparison
rs = RatingSpaceComparison(rating_matrix, users)
rs.movies = movies
rs.filter_movie()
rs_result = rs.result
print(rs_result)
data_output = open('rs_result.pkl','wb')
pickle.dump(rs_result,data_output)
data_output.close()

# BN
# bn = BN(rating_matrix, users, 'chow-liu')
# bn.filter_movie()
