from sklearn.model_selection import KFold
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np

class RatingSpaceComparison:
    def __init__(self, rating_matrix, demographic):
        self.RM = rating_matrix
        self.spliter = KFold(n_splits=5)
        self.D = demographic
        self.result = {}

    def test(self, X, y, type:str):
        scores = []
        f1s = []
        reports = []
        for train_ids, test_ids in self.spliter.split(X):
            x_train, x_test = X.iloc[train_ids], X.iloc[test_ids]
            y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]

            # train_df = X.iloc[train_ids]
            y_pred = []
            for user_id in x_test.index:
                ratings = list(x_test.loc[user_id])
                df = x_train.apply(lambda x: 1 - cosine(x, ratings), axis=1)
                match_user = df.nlargest(1).index[0]
                y_pred.append(y_train.loc[match_user])
            y_true = list(y.iloc[test_ids])
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            scores.append(accuracy)
            f1s.append(f1)
            reports.append(classification_report(y_true, y_pred))
            # print('accuracy: ', accuracy_score(y_true, y_pred))
            # print('f1: ', f1_score(y_true, y_pred, average="weighted"))
            # print('report: ', classification_report(y_true, y_pred))
        # print('max accuracy of ', type, ' prediction is: ', max(scores))
        max_index = max(enumerate(scores), key=lambda x:x[1])[0]
        print('max accuracy of ', type, ' prediction\'s model is: ')
        print(reports[max_index])
        return np.mean(scores), np.mean(f1s)

    def filter_movie(self):
        for num_movies in range(5, 21, 3):
            print("number of movies:" , num_movies)
            self.result[num_movies] = {}
            for mode in ('random', 'anova'):
                self.result[num_movies][mode] = {}
                print('select movies with {} method'.format(mode))
                if mode == 'random':
                    X = self.RM.sample(n=num_movies, axis=1)
                else:
                    selected_movies = self.movies[:num_movies]
                    X = self.RM[selected_movies]
                X = pd.merge(X, self.D[['gender', 'age', 'occupation']], on='user_id')
                X = X[X.astype(bool).sum(axis=1)>3]
                y_gender = X.iloc[: , -3]
                y_age = X.iloc[:, -2]
                y_occu = X.iloc[:, -1]
                X = X.drop(columns=['gender', 'age', 'occupation'])

                self.result[num_movies][mode]['gender'] = self.test(X, y_gender, 'gender')
                self.result[num_movies][mode]['age'] = self.test(X, y_age, 'age')
                self.result[num_movies][mode]['occu'] = self.test(X, y_occu, 'occu')


