import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


from basic_model.model import Model

class LR(Model):
    def __init__(self, rating_matrix, demographic):
        super().__init__(rating_matrix, demographic)
        self.models = {
            'gender': LogisticRegression(solver='liblinear', C=0.05),
            'age': LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.05, max_iter=4000),
            'occupation': LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.05, max_iter=4000)
        }


    def train(self, X, num_movies):
        # X = X.values
        print('total users: {}'.format(len(X)))
        for prop in self.groups:
            y = self.D

            for user_type in ('critic', 'active'):
                for more_or_less in ('more', 'less'):
                    print(num_movies, prop, user_type, more_or_less)
                    y_pred = {'gender': [], 'age': [], 'occupation': []}
                    for train_ids, test_ids in self.spliter.split(X):
                        x_train, x_test = X.iloc[train_ids], X.iloc[test_ids]
                        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]

                        x_train = self.get_users(x_train, prop, user_type, more_or_less)
                        y_train = y_train.loc[x_train.index]
                        for type in ('gender', 'age', 'occupation'):
                            model = self.models[type]
                            model.fit(x_train.to_numpy(), y_train[type].to_numpy())
                            y_pred[type].extend(list(model.predict(x_test.to_numpy())))
                            # predict.extend(y_pred)
                    predict = list(zip(y_pred['gender'], y_pred['age'], y_pred['occupation']))
                    self.evaluate(predict, num_movies, prop, user_type, more_or_less)


    def filter_movie(self):
        for num_movies in self.movies_group:
            print("number of movies:" , num_movies)
            for mode in ('random', 'anova'):
                if mode == 'anova':
                    continue
                print('select movies with {} method'.format(mode))
                if mode == 'random':
                    X = self.RM.sample(n=num_movies, axis=1)
                    while sum(X.astype(bool).sum(axis=1)==0) > num_movies * 0.5:
                        X = self.RM.sample(n=num_movies, axis=1)
                else:
                    selected_movies = self.movies[:num_movies]
                    X = self.RM[selected_movies]
                # X = pd.merge(X, self.D[['gender', 'age', 'occupation']], on='user_id')
                # X = X[X.astype(bool).sum(axis=1)>3]
                # X = X.drop(columns=['gender', 'age', 'occupation'])

                self.train(X, num_movies)

