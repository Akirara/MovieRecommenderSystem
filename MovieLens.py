import os
import csv
import datetime
import heapq
from operator import itemgetter
from collections import defaultdict


def load_reviews(path, **kwargs):
    """
    Load MovieLens reviews
    """
    options = {
        'fieldnames': ('userid', 'movieid', 'rating', 'timestamp'),
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
            row['userid'] = parse_int(row, 'userid')
            row['movieid'] = parse_int(row, 'movieid')
            row['rating'] = parse_int(row, 'rating')
            row['timestamp'] = parse_date(row, 'timestamp')
            yield row


def relative_path(path):
    """
    Return a relative path from the file
    """
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname, path)
    return os.path.normpath(path)


def load_movies(path, **kwargs):
    """
    Load MovieLens movies
    """
    options = {
        'fieldnames': ('movieid', 'title', 'release', 'video', 'url'),
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
            row['movieid'] = parse_int(row, 'movieid')
            row['release'] = parse_date(row, 'release')
            row['video'] = parse_date(row, 'video')
            yield row


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
            self.movies[movie['movieid']] = movie

        for review in load_reviews(self.udata):
            self.reviews[review['userid']][review['movieid']] = review

    def reviews_for_movie(self, movieid):
        """
        Yield the reviews for a given movie
        """
        for review in self.reviews.values():
            if movieid in review:
                yield review[movieid]

    def average_reviews(self):
        """
        Average the star rating for all movies
        Yield a tuple of movieid, the average rating and the number of reviews
        """
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = sum(reviews) / float(len(reviews))
            yield (movieid, average, len(reviews))

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
        for movieid in self.movies:
            reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
            average = ((c * m) + sum(reviews)) / float(c + len(reviews))
            yield (movieid, average, len(reviews))

    def bayesian_top_rated(self, n=10, c=59, m=3):
        """
        Yield the n top rated movie based on Bayesian average
        """
        return heapq.nlargest(n, self.bayesian_average(c, m), key=itemgetter(1))


if __name__ == "__main__":
    data = relative_path('ml-100k/u.data')
    item = relative_path('ml-100k/u.item')
    model = MovieLens(data, item)

    for mid, avg, num in model.bayesian_top_rated(10):
        title = model.movies[mid]['title']
        print("[%0.3f average rating (%i reviews)] %s" % (avg, num, title))
