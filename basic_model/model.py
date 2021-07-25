from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

class Model:
    def __init__(self, rating_matrix, demographic):
        self.RM = rating_matrix
        self.D = demographic
        self.movies_group = [int(i * 0.01 * len(self.RM.columns)) for i in range(5, 101, 5)]
        self.user_groups = {}
        self.groups = [i/80 for i in range(0, 81, 5)]
        self.groups[0] = 1/80
        self.users = {
            'critic': {
                'more': self.RM.apply(self.get_user_var, axis=1).sort_values(ascending=False).index,
                'less': self.RM.apply(self.get_user_var, axis=1).sort_values(ascending=True).index
            },
            'active': {
                'more': self.RM.apply(self.get_user_rating_count, axis=1).sort_values(ascending=False).index,
                'less': self.RM.apply(self.get_user_rating_count, axis=1).sort_values(ascending=True).index
            }

        }
        self.spliter = KFold(n_splits=5)
        self.reports = []

    def get_user_rating_count(self, user):
        return user.astype(bool).sum()

    def get_user_var(self, user):
        user = user[user != 0]
        return user.var()

    def get_users(self, df, prop, user_type, more_or_less):
        num_user = int(len(df) * prop)
        df = df.reindex(self.users[user_type][more_or_less]).dropna()
        return df.iloc[: num_user]

    def evaluate(self, predict, num_movies, prop, user_type, more_or_less):
        for i, attr in enumerate(['gender', 'age', 'occupation']):
            y_pred, y_test = list(map(lambda x:x[i], predict)), list(self.D[attr])
            self.reports.append(
                [user_type, more_or_less, num_movies, attr, prop * 80, classification_report(y_test, y_pred, zero_division=0, output_dict=True)])

