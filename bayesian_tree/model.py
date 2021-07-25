import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pomegranate import BayesianNetwork
from sklearn.metrics import accuracy_score, f1_score, classification_report
from basic_model.model import Model

class BN(Model):
    def __init__(self, rating_matrix, demographic, algo='chow-liu'):
        super().__init__(rating_matrix, demographic)
        self.algo = algo

    def train(self, df, num_movies):
        reports = []
        print('total users: {}'.format(len(df)))
        for prop in self.groups:
            predict = []
            for train_ids, test_ids in self.spliter.split(df):

                train_df, test_df = df.iloc[train_ids], df.iloc[test_ids]

                train_df = self.get_users(train_df, prop, 'critic', 'more')

                for i in range(len(test_df)):
                    ratings = test_df.iloc[i][:-3]
                    valid_ratings = list(ratings[ratings != 0].index)[:num_movies]
                    X = train_df[valid_ratings+['gender', 'age', 'occupation']]
                    X = X[X.astype(bool).sum(axis=1)>3]

                    #######################################################
                    # set up edges constraint on exact and greedy algorithm
                    included_edges = []
                    excluded_edges = []
                    gender_node = num_movies
                    age_node = num_movies + 1
                    occu_node = num_movies + 2
                    for index in range(num_movies):
                        included_edges.extend([(index, occu_node), (index, age_node), (index, gender_node)])
                        excluded_edges.extend([(gender_node, index), (age_node, index), (occu_node, index)])
                    ########################################################

                    if self.algo == 'chow-liu':
                        try:
                            model = BayesianNetwork.from_samples(np.array(X), n_jobs=-1, pseudocount=1, algorithm='chow-liu')
                        except:
                            print(X)
                            print(num_movies, prop)
                    else:
                        model = BayesianNetwork.from_samples(np.array(X), n_jobs=-1, pseudocount=1, exclude_edges=excluded_edges, algorithm=self.algo)
                    try:
                        pred = model.predict([list(ratings[valid_ratings]) + [None, None, None]])
                        predict.append(list(pred[0])[-3:])
                    except Exception as e:
                        predict.append(['M', 25, 4])
            for i, attr in enumerate(['gender', 'age', 'occupation']):
                y_pred, y_test = list(map(lambda x:x[i], predict)), list(self.D[attr])
                self.reports.append([num_movies, attr, prop*80, classification_report(y_test, y_pred, zero_division=0, output_dict=True)])
                for item in self.reports[-1]:
                    print(item)
        return reports

    def filter_movie(self):


        for num_movies in range(5, 21, 3):
            print("number of movies:", num_movies)
            X = pd.merge(self.RM, self.D[['gender', 'age', 'occupation']], on='user_id')
            self.train(X, num_movies)
