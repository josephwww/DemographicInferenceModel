import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import KFold

class LR:
    def __init__(self, rating_matrix, demographic):
        self.models = {
            'gender': LogisticRegression(solver='liblinear', C=0.05),
            'age': LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.05, max_iter=4000),
            'occu': LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.05, max_iter=4000)
        }
        self.RM = rating_matrix
        self.D = demographic
        self.spliter = KFold(n_splits=5)
        self.result = {}

    def train(self, X, y, type:str):
        X = X.values
        y = y.to_numpy()
        scores = []
        reports = []
        print('total users: {}'.format(len(X)))
        for train_ids, test_ids in self.spliter.split(X):
            x_train, x_test = X[train_ids], X[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]
            model = self.models[type]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            # print('report: ')
            reports.append(classification_report(y_test, y_pred, zero_division=0))
            scores.append(model.score(x_test, y_test))

        max_index = max(enumerate(scores), key=lambda x:x[1])[0]
        print('max accuracy of ', type, ' prediction\'s model is: ')
        print(reports[max_index])
        return np.mean(scores)

    def filter_movie(self):
        for num_movies in range(5, 16, 3):
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

                self.result[num_movies][mode]['gender'] = self.train(X, y_gender, 'gender')
                self.result[num_movies][mode]['age'] = self.train(X, y_age, 'age')
                self.result[num_movies][mode]['occu'] = self.train(X, y_occu, 'occu')

    # def remove_duplicate_movies(self):
    #     temp = []
    #     for movie in self.movies:
    #         if movie not in temp:
    #             temp.append(movie)
    #     self.movies = temp

