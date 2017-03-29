import os
import csv
import datetime
import heapq
from math import sqrt
from operator import itemgetter
from collections import defaultdict


def load_reviews(path, **kwargs):
    """
    Load MovieLens reviews
    """
    options = {
        'fieldnames': ('user_id', 'movie_id', 'rating', 'timestamp'),
        'delimiter': '\t'
    }
    options.update(kwargs)
    
    def parse_date(r, k):
        return datetime.datetime.fromtimestamp(float(r[k]))
    
    def parse_int(r, k):
        return int(r[k])

    with open(path, 'r', encoding='utf-8') as reviews:
        reader = csv.DictReader(reviews, **options)
        for row in reader:
            row['user_id'] = parse_int(row, 'user_id')
            row['movie_id'] = parse_int(row, 'movie_id')
            row['rating'] = parse_int(row, 'rating')
            row['timestamp'] = parse_date(row, 'timestamp')
            yield row


def load_movies(path, **kwargs):
    """
    Load MovieLens movies
    """
    options = {
        'fieldnames': ('movie_id', 'title', 'release', 'video', 'url'),
        'delimiter': '|',
        'restkey': 'genre'
    }
    options.update(kwargs)

    def parse_int(r, k):
        return int(r[k])

    def parse_date(r, k):
        return datetime.datetime.strptime(r[k], '%d-%b-%Y') if r[k] else None

    with open(path, 'r', encoding='utf-8') as movies:
        reader = csv.DictReader(movies, **options)
        for row in reader:
            row['movie_id'] = parse_int(row, 'movie_id')
            row['release'] = parse_date(row, 'release')
            row['video'] = parse_date(row, 'video')
            yield row


def relative_path(path):
    """
    Return a relative path from the file
    """
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname, path)
    return os.path.normpath(path)


class MovieLens(object):
    """
    Data structure to build the recommender model on
    """

    def __init__(self, udata, uitem):
        """
        Instantiate with a path to u.data and u.item
        """
        self.udata = udata
        self.uitem = uitem
        self.movies = {}
        self.reviews = defaultdict(dict)
        self.load_dataset()

    def load_dataset(self):
        """
        Load the two datasets into memory, indexed on the ID
        """
        for movie in load_movies(self.uitem):
            self.movies[movie['movie_id']] = movie

        for review in load_reviews(self.udata):
            self.reviews[review['user_id']][review['movie_id']] = review

    def reviews_for_movie(self, movie_id):
        """
        Yield the reviews for a given movie
        """
        for review in self.reviews.values():
            if movie_id in review:
                yield review[movie_id]

    def average_reviews(self):
        """
        Average the star rating for all movies
        Yield a tuple of movie_id, the average rating and the number of reviews
        """
        for movie_id in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movie_id))
            average = sum(reviews) / float(len(reviews))
            yield (movie_id, average, len(reviews))

    def top_rated(self, n=10):
        """
        Yield the n top rated movies
        """
        return heapq.nlargest(n, self.average_reviews(), key=itemgetter(1))

    def bayesian_average(self, c, m):
        """
        Report the Bayesian average with parameters c and m
        c: threshold of reviews number, default = average reviews for each movie
        m: threshold of rating
        """
        for movie_id in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movie_id))
            average = ((c * m) + sum(reviews)) / float(c + len(reviews))
            yield (movie_id, average, len(reviews))

    def bayesian_top_rated(self, n=10, c=59, m=3):
        """
        Yield the n top rated movie based on Bayesian average
        """
        return heapq.nlargest(n, self.bayesian_average(c, m), key=itemgetter(1))

    def shared_preferences(self, critic_a, critic_b):
        """
        Return the intersection of ratings for two critics
        """
        if critic_a not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data" % critic_a)
        if critic_b not in self.reviews:
            raise KeyError("Couldn't find critic '%s' in data" % critic_b)

        movies_a = set(self.reviews[critic_a].keys())
        movies_b = set(self.reviews[critic_b].keys())
        shared = movies_a & movies_b

        reviews = {}
        for movie_id in shared:
            reviews[movie_id] = (
                self.reviews[critic_a][movie_id]['rating'],
                self.reviews[critic_b][movie_id]['rating']
            )
        return reviews

    def euclidean_distance(self, a, b, prefs='critics'):
        """
        Return the Euclidean distance of two critics or two movies, A&B by performing a J-dimensional
        Euclidean calculation of each of their preference vectors for the intersection of movies the
        critics have rated
        """
        if prefs == 'critics':
            preferences = self.shared_preferences(a, b)
        elif prefs == 'movies':
            preferences = self.shared_critics(a, b)
        else:
            raise KeyError("Unknown preference type: '%s'" % prefs)

        if len(preferences) == 0:
            return 0

        sum_of_squares = sum([pow(a - b, 2) for a, b in preferences.values()])

        # add 1 to prevent division by zero error
        return 1 / (1 + sqrt(sum_of_squares))

    def pearson_correlation(self, a, b, prefs='critics'):
        """
        Return the Pearson Correlation of two critics or two movies, A&B by performing the PPMC calculation on
        the scatter plot of (a, b) ratings on the shared set of critiqued titles
        """
        if prefs == 'critics':
            preferences = self.shared_preferences(a, b)
        elif prefs == 'movies':
            preferences = self.shared_critics(a, b)
        else:
            raise KeyError("Unknown preference type: '%s'" % prefs)

        length = len(preferences)

        if length == 0:
            return 0

        sum_a = sum_b = sum_square_a = sum_square_b = sum_product = 0
        for a, b in preferences.values():
            sum_a += a
            sum_b += b
            sum_square_a += pow(a, 2)
            sum_square_b += pow(b, 2)
            sum_product += a * b

        numerator = (sum_product * length) - (sum_a * sum_b)
        denominator = sqrt(((sum_square_a * length) - pow(sum_a, 2)) * ((sum_square_b * length) - pow(sum_b, 2)))

        if denominator == 0:
            return 0

        return abs(numerator / denominator)

    def similar_critics(self, user, metric='euclidean', n=None):
        """
        Find and rank similar critics for the user according to the specified distance metric
        Return the top n similar critics if n is specified
        """
        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation
        }
        distance = metrics.get(metric, None)

        if user not in self.reviews:
            raise KeyError("Unknown user, '%s'." % user)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unrealized distance metric '%s'." % metric)

        critics = {}
        for critic in self.reviews:
            if critic != user:
                critics[critic] = distance(user, critic)

        if n:
            return heapq.nlargest(n, critics.items(), key=itemgetter(1))
        return critics

    def predict_ranking_user_based(self, user, movie, metric='euclidean', critics=None):
        """
        Predict the ranking a user might give a movie based on the weighted average of the
        critics similar to the user
        """
        critics = critics or self.similar_critics(user, metric=metric)
        total = 0.0
        sim_sum = 0.0

        for critic, similarity in critics.items():
            if movie in self.reviews[critic]:
                total += similarity * self.reviews[critic][movie]['rating']
                sim_sum += similarity

        if sim_sum == 0.0:
            return 0.0
        return total / sim_sum

    def predict_all_rankings(self, user, metric='euclidean', n=None):
        """
        Predict all rankings for all movies, if n is specified return the top
        n movies and their predicted ranking
        """
        critics = self.similar_critics(user, metric=metric)
        movies = {
            movie: self.predict_ranking_user_based(user, movie, metric, critics) for movie in self.movies
        }

        if n:
            return heapq.nlargest(n, movies.items(), key=itemgetter(1))
        return movies

    def shared_critics(self, movie_a, movie_b):
        """
        Return the intersection of critics for two items, A&B
        """
        if movie_a not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" % movie_a)
        if movie_b not in self.movies:
            raise KeyError("Couldn't find movie '%s' in data" % movie_b)

        critic_a = set(critic for critic in self.reviews if movie_a in self.reviews[critic])
        critic_b = set(critic for critic in self.reviews if movie_b in self.reviews[critic])
        shared = critic_a & critic_b

        reviews = {}
        for critic in shared:
            reviews[critic] = (
                self.reviews[critic][movie_a]['rating'],
                self.reviews[critic][movie_b]['rating']
            )
        return reviews

    def similar_items(self, movie, metric='euclidean', n=None):
        metrics = {
            'euclidean': self.euclidean_distance,
            'pearson': self.pearson_correlation
        }

        distance = metrics.get(metric, None)

        if movie not in self.reviews:
            raise KeyError("Unknown movie, '%s'." % movie)
        if not distance or not callable(distance):
            raise KeyError("Unknown or unrealized distance metric '%s'" % metric)

        items = {}
        for item in self.movies:
            if item != movie:
                items[item] = distance(item, movie, prefs='movies')

        if n:
            return heapq.nlargest(n, items.items(), key=itemgetter(1))
        return items

    def predict_ranking_item_based(self, user, movie, metric='euclidean'):
        movies = self.similar_items(movie, metric=metric)
        total = 0.0
        sim_sum = 0.0

        for rel_movie, similarity in movies.items():
            if rel_movie in self.reviews[user]:
                total += similarity * self.reviews[user][rel_movie]['rating']
                sim_sum += similarity

        if sim_sum == 0.0:
            return 0.0
        return total / sim_sum


if __name__ == "__main__":
    data = relative_path('ml-100k/u.data')
    item = relative_path('ml-100k/u.item')
    model = MovieLens(data, item)

    # for mid, avg, num in model.bayesian_top_rated(10):
    #     title = model.movies[mid]['title']
    #     print("[%0.3f average rating (%i reviews)] %s" % (avg, num, title))

    # for item in model.similar_critics(232, 'euclidean', n=10):
    #     print("%4i: %0.3f" % item)

    # for mid, rating in model.predict_all_rankings(578, 'pearson', 10):
    #     print("%0.3f: %s" % (rating, model.movies[mid]['title']))

    # for movie, similarity in model.similar_items(631, 'pearson').items():
    #     print("%0.3f: %s" % (similarity, model.movies[movie]['title']))
