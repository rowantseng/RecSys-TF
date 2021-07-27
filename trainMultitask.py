import json
import os

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from sequentials import buildVectorizer, lookupEmbSequentialModel
from train import MovieModel, UserModel, train, evaluate


class MovielensMultiLayerModel(tfrs.models.Model):
    def __init__(self, candidates, users, movies, maxWord,
                 vectorizer, nFeature=32,
                 queryLayers=[32], candidateLayers=[32], ratingLayers=[1],
                 retrievalWeight=1., ratingWeight=1.):
        super().__init__()
        self.queryDenseLayers = tf.keras.Sequential()
        for ls in queryLayers[:-1]:
            self.queryDenseLayers.add(
                tf.keras.layers.Dense(ls, activation="relu"))
        self.queryDenseLayers.add(tf.keras.layers.Dense(queryLayers[-1]))

        self.candidateDenseLayers = tf.keras.Sequential()
        for ls in candidateLayers[:-1]:
            self.candidateDenseLayers.add(
                tf.keras.layers.Dense(ls, activation="relu"))
        self.candidateDenseLayers.add(
            tf.keras.layers.Dense(candidateLayers[-1]))

        self.ratingModel = tf.keras.Sequential()
        for ls in ratingLayers[:-1]:
            self.ratingModel.add(
                tf.keras.layers.Dense(ls, activation="relu"))
        self.ratingModel.add(
            tf.keras.layers.Dense(ratingLayers[-1]))

        self.queryModel = tf.keras.Sequential([
            UserModel(users=users, nFeature=nFeature),
            self.queryDenseLayers
        ])
        self.candidateModel = tf.keras.Sequential([
            MovieModel(movies=movies, maxWord=maxWord,
                       vectorizer=vectorizer, nFeature=nFeature),
            self.candidateDenseLayers
        ])

        self.retrievalTask = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates.batch(128).map(self.candidateModel),
            ),
        )
        self.ratingTask = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        self.ratingWeight = ratingWeight
        self.retrievalWeight = retrievalWeight

    def call(self, features):
        queryEmbeddings = self.queryModel({"user_id": features["user_id"]})
        movieEmbeddings = self.candidateModel(features["movie_title"])
        ratings = self.ratingModel(
            tf.concat([queryEmbeddings, movieEmbeddings], axis=1))
        return queryEmbeddings, movieEmbeddings, ratings

    def compute_loss(self, features, training=False):
        ratingLabels = features.pop("user_rating")
        queryEmbeddings, movieEmbeddings, ratings = self(features)
        ratingLoss = self.ratingTask(ratingLabels, ratings)
        retrievalLoss = self.retrievalTask(queryEmbeddings, movieEmbeddings)
        totalLoss = self.ratingWeight * ratingLoss + \
            self.retrievalWeight * retrievalLoss
        return totalLoss


if __name__ == "__main__":
    # Load data
    ratings = tfds.load("movielens/100k-ratings", split="train")
    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })

    # Define movie title vectorizer
    maxWord = 10000
    movies = movies.map(lambda x: x["movie_title"])
    vectorizer = buildVectorizer(x=movies, maxWord=maxWord)

    # Define number of user and movie
    uniqueMovieTitles = np.unique(np.concatenate(list(movies.batch(1000))))
    uniqueUserIds = np.unique(np.concatenate(list(ratings.batch(1000).map(
        lambda x: x["user_id"]))))

    # Define params
    seed = 42
    lr = 0.01
    epochs = 100
    trainNum, validNum = 80000, 20000
    totalNum = trainNum + validNum
    trainBsz = 2048
    validBsz = 4096
    params = {"candidates": movies, "users": uniqueUserIds, "movies": uniqueMovieTitles,
              "maxWord": maxWord, "vectorizer": vectorizer, "nFeature": 32,
              "queryLayers": [32], "candidateLayers": [32], "ratingLayers": [64, 32, 1],
              "ratingWeight": 1., "retrievalWeight": 1.}

    # Generate Data
    tf.random.set_seed(seed)
    shuffled = ratings.shuffle(
        totalNum, seed=seed, reshuffle_each_iteration=False)

    trainSet = shuffled.take(trainNum)
    validSet = shuffled.skip(trainNum).take(validNum)
    trainSet = trainSet.shuffle(totalNum).batch(trainBsz).cache()
    validSet = validSet.batch(validBsz).cache()

    # Train and eval on retrieval task only
    params["ratingWeight"] = 0.
    params["retrievalWeight"] = 1.
    resultPath = "checkpoints/multitask_retrieval"
    model = MovielensMultiLayerModel(**params)
    model = train(trainSet, validSet, model, epochs, lr, resultPath)
    evaluate(validSet, movies, model, resultPath)

    # Train and eval on rating and retrieval tasks
    params["ratingWeight"] = 1.
    params["retrievalWeight"] = 1.
    resultPath = "checkpoints/multitask_rating_retrieval"
    model = MovielensMultiLayerModel(**params)
    model = train(trainSet, validSet, model, epochs, lr, resultPath)
    evaluate(validSet, movies, model, resultPath)
