# Movie Recommender System
A simple movie recommender system based on MovieLens dataset

MovieLens dataset: https://grouplens.org/datasets/movielens/

## Introduction
The movie recommender system implements functions such as User-Based CF, Item-Based CF and SVD module

### movie_lens
The MovieLens class provides:

&#8195;(1)Average ratings calculation with normal and bayesian method

&#8195;(2)Euclidean distance and Pearson correlation of movies/users

&#8195;(3)Ranking prediction

### recommender
The Recommender class implements prediction model based on SVD and model persistence

### svd
Implement matrix factorization using gradient descent
