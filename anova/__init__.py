import pandas as pd
import scipy.stats as stats

class Anova():
    def __init__(self, rating_matrix, D):
        self.ratings = rating_matrix
        self.users = D
        self.gender_list = self.get_gender_list()
        self.age_list = self.get_age_list()
        self.occupation_list = self.get_occupation_list()
        self.movies = [movie for movies in list(zip(self.gender_list, self.age_list, self.occupation_list)) for movie in movies]
        self.remove_duplicate_movies(self.movies)

    def get_gender_list(self):
        gender = self.users[['gender']]
        age = self.users[['age']]
        result = []
        anova_gender = []
        for movie_id in self.ratings.columns:
            movie = self.ratings[movie_id]
            if (len(movie[movie != 0]) < 50):
                anova_gender.append(movie_id)
                continue
            df = pd.concat([movie, gender], axis=1)
            groups = df.groupby("gender").groups
            male = df[movie_id][groups['M']]
            male = male[male != 0]
            female = df[movie_id][groups['F']]
            female = female[female != 0]
            male.reset_index(drop=True, inplace=True)
            female.reset_index(drop=True, inplace=True)

            statistic, pvalue = stats.f_oneway(male, female)
            result.append((movie_id, statistic, pvalue))
        anova_gender = list(map(lambda x: x[0], sorted(result, key=lambda x: x[1], reverse=True))) + anova_gender
        return anova_gender

    def get_age_list(self):
        age = self.users[['age']]
        age_result = []
        anova_age = []
        for movie_id in self.ratings.columns:
            movie = self.ratings[movie_id]
            if (len(movie[movie != 0]) < 50):
                anova_age.append(movie_id)
                continue
            df = pd.concat([movie, age], axis=1)
            df = df[df[movie_id] != 0]
            groups = df.groupby("age").groups
            if (len(groups) < 7):
                anova_age.append(movie_id)
                continue
            _1 = df[movie_id][groups[1]]
            _18 = df[movie_id][groups[18]]
            _25 = df[movie_id][groups[25]]
            _35 = df[movie_id][groups[35]]
            _45 = df[movie_id][groups[45]]
            _50 = df[movie_id][groups[50]]
            _56 = df[movie_id][groups[56]]

            statistic, pvalue = stats.f_oneway(_1, _18, _25, _35, _45, _50, _56)
            age_result.append((movie_id, statistic, pvalue))
            # print(age_result[-1])
        anova_age = list(map(lambda x: x[0], sorted(age_result, key=lambda x: x[1], reverse=True))) + anova_age
        return anova_age

    def get_occupation_list(self):
        occupation = self.users[['occupation']]
        occupation_result = []
        anova_occupation = []
        for movie_id in self.ratings.columns:
            movie = self.ratings[movie_id]
            if (len(movie[movie != 0]) < 50):
                anova_occupation.append(movie_id)
                continue
            df = pd.concat([movie, occupation], axis=1)
            df = df[df[movie_id] != 0]
            groups = df.groupby("occupation").groups
            if (len(groups) < 21):
                anova_occupation.append(movie_id)
                continue
            occu_group = []
            for occu_id in range(21):
                occu_group.append(df[movie_id][groups[occu_id]])

            statistic, pvalue = stats.f_oneway(*occu_group)
            occupation_result.append((movie_id, statistic, pvalue))
            # print(occupation_result[-1])
        anova_occupation = list(map(lambda x: x[0], sorted(occupation_result, key=lambda x: x[1], reverse=True))) + anova_occupation
        return anova_occupation

    def remove_duplicate_movies(self, movies):
        temp = []
        for movie in movies:
            if movie not in temp:
                temp.append(movie)
        self.movies = temp
