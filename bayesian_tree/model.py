import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pomegranate import BayesianNetwork
from sklearn.metrics import accuracy_score, f1_score, classification_report

class BN:
    def __init__(self, rating_matrix, demographic):
        self.RM = rating_matrix
        self.D = demographic
        self.spliter = KFold(n_splits=5)
        self.result = {}

    def train(self, df, num_movies):
        scores = []
        reports = []
        print('total users: {}'.format(len(df)))
        for train_ids, test_ids in self.spliter.split(df):

            predict=[]
            train_df, test_df = df.iloc[train_ids], df.iloc[test_ids]
            D_train, D_test = self.D.iloc[train_ids], self.D.iloc[test_ids]

            for i in range(len(test_df)):
                # print(i)
                ratings = test_df.iloc[i][:-3]
                valid_ratings = list(ratings[ratings != 0].index)[:num_movies]
                X = train_df[valid_ratings+['gender', 'age', 'occupation']]
                X = X[X.astype(bool).sum(axis=1)>3]
                included_edges = []
                excluded_edges = []
                gender_node = num_movies
                age_node = num_movies + 1
                occu_node = num_movies + 2
                for index in range(num_movies):
                    included_edges.extend([(index, occu_node), (index, age_node), (index, gender_node)])
                    excluded_edges.extend([(gender_node, index), (age_node, index), (occu_node, index)])
                model = BayesianNetwork.from_samples(np.array(X), n_jobs=-1, pseudocount=1, include_edges=included_edges, exclude_edges=excluded_edges)
                try:
                    pred = model.predict([list(ratings[valid_ratings]) + [None, None, None]])
                    predict.append(list(pred[0])[-3:])
                except:
                    predict.append(['M', 25, 4])
                # print(predict[-1])
            # print('report: ')

            for i, attr in enumerate(['gender', 'age', 'occupation']):
                y_pred, y_test = map(lambda x:x[i], predict), list(D_test[attr])
                reports.append(classification_report(y_test, y_pred, zero_division=0))
                scores.append(accuracy_score(y_test, y_pred))

        max_index = max(enumerate(scores), key=lambda x:x[1])[0]
        print('max accuracy\'s model is: ')
        print(reports[max_index])
        print(reports[max_index+1])
        print(reports[max_index+2])

    def filter_movie(self):
        for num_movies in range(5, 16, 3):
            print("number of movies:" , num_movies)
            self.result[num_movies] = {}
            X = pd.merge(self.RM, self.D[['gender', 'age', 'occupation']], on='user_id')

            self.train(X, num_movies)

