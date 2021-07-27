import json
import os
import tempfile
import time

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from sequentials import (buildNormalizer, buildVectorizer,
                         lookupEmbSequentialModel, qdEmbSequentialModel)


class UserModel(tf.keras.Model):
    def __init__(self, users, useTimestamp=False, normalizer=None, qdBucket=None, nFeature=32):
        super().__init__()
        self.useTimestamp = useTimestamp
        if useTimestamp and not normalizer:
            raise AssertionError(
                "Please assign 'normalizer' for timestamp normalization!")
        if useTimestamp and not qdBucket:
            raise AssertionError(
                "Please assign 'qdBucket' for discretization!")

        self.userEmbedding = lookupEmbSequentialModel(
            vocabs=users, nFeature=nFeature)

        if useTimestamp:
            self.timeEmbedding = qdEmbSequentialModel(
                buckets=qdBucket, nFeature=nFeature)
            self.normalizer = normalizer

    def call(self, inputs):
        if not self.useTimestamp:
            return self.userEmbedding(inputs["user_id"])

        return tf.concat([
            self.userEmbedding(inputs["user_id"]),
            self.timeEmbedding(inputs["timestamp"]),
            self.normalizer(inputs["timestamp"]),
        ], axis=1)


class MovieModel(tf.keras.Model):
    def __init__(self, movies, maxWord=10000, vectorizer=None, nFeature=32):
        super().__init__()
        if not vectorizer:
            raise AssertionError(
                "Please assign 'vectorizer' for title transformation!")

        self.titleEmbedding = lookupEmbSequentialModel(
            vocabs=movies, nFeature=nFeature)

        self.titleWordEmbedding = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(maxWord, nFeature, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

    def call(self, titles):
        return tf.concat([
            self.titleEmbedding(titles),
            self.titleWordEmbedding(titles),
        ], axis=1)


class MovielensMultiLayerModel(tfrs.models.Model):
    def __init__(self, candidates, users, movies, maxWord, normalizer,
                 vectorizer, qdBucket, useTimestamp=False, nFeature=32,
                 queryLayers=[32], candidateLayers=[32]):
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

        self.queryModel = tf.keras.Sequential([
            UserModel(users=users, useTimestamp=useTimestamp,
                      normalizer=normalizer, qdBucket=qdBucket, nFeature=nFeature),
            self.queryDenseLayers
        ])
        self.candidateModel = tf.keras.Sequential([
            MovieModel(movies=movies, maxWord=maxWord,
                       vectorizer=vectorizer, nFeature=nFeature),
            self.candidateDenseLayers
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates.batch(128).map(self.candidateModel),
            ),
        )

    def compute_loss(self, features, training=False):
        queryEmbeddings = self.queryModel({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
        })
        movieEmbeddings = self.candidateModel(features["movie_title"])

        return self.task(queryEmbeddings, movieEmbeddings)


def train(trainSet, validSet, model, epochs, lr, resultPath, validFreq=5):
    model.compile(optimizer=tf.keras.optimizers.Adagrad(lr))
    history = model.fit(trainSet, validation_data=validSet,
                        validation_freq=validFreq, epochs=epochs)

    # Save results
    os.makedirs(resultPath, exist_ok=True)
    json.dump(history.history, open(f"{resultPath}/history.json", "w"))
    return model


def evaluate(validSet, candidates, model, resultPath):
    movies = candidates.batch(4096)
    movieEmbeddings = candidates.batch(4096).map(model.candidateModel)

    bruteForceAlgo = tfrs.layers.factorized_top_k.BruteForce(model.queryModel)
    bruteForceAlgo.index(movieEmbeddings, movies)

    scannAlgo = tfrs.layers.factorized_top_k.ScaNN(model.queryModel)
    scannAlgo.index(movieEmbeddings, movies)

    scannOptimAlgo = tfrs.layers.factorized_top_k.ScaNN(
        model.queryModel, num_leaves=1000, num_leaves_to_search=100, num_reordering_candidates=1000)
    scannOptimAlgo.index(movieEmbeddings, movies)

    # This is used for defining input shape
    _ = bruteForceAlgo({"user_id": np.array(["42"])})
    _ = scannAlgo({"user_id": np.array(["42"])})
    _ = scannOptimAlgo({"user_id": np.array(["42"])})

    # Write out
    with tempfile.TemporaryDirectory() as tmp:
        bruteForceAlgo.save(f"{resultPath}/bruteforce")
        scannAlgo.save(f"{resultPath}/scann",
                       options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))
        scannOptimAlgo.save(f"{resultPath}/optimScann",
                            options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))


if __name__ == "__main__":
    # Load data
    ratings = tfds.load("movielens/100k-ratings", split="train")
    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "timestamp": x["timestamp"],
    })

    # Define movie title vectorizer
    maxWord = 10000
    movies = movies.map(lambda x: x["movie_title"])
    vectorizer = buildVectorizer(x=movies, maxWord=maxWord)

    # Define timestamp discretizer
    timestamps = np.concatenate(
        list(ratings.map(lambda x: x["timestamp"]).batch(100)))
    qdBucket = np.linspace(
        timestamps.min(), timestamps.max(), num=1000).tolist()
    normalizer = buildNormalizer(timestamps)

    # Define number of user and movie
    uniqueMovieTitles = np.unique(np.concatenate(list(movies.batch(1000))))
    uniqueUserIds = np.unique(np.concatenate(list(ratings.batch(1000).map(
        lambda x: x["user_id"]))))

    # Define params
    seed = 42
    lr = 0.1
    epochs = 200
    trainNum, validNum = 80000, 20000
    totalNum = trainNum + validNum
    trainBsz = 2048
    validBsz = 4096
    params = {"candidates": movies, "users": uniqueUserIds, "movies": uniqueMovieTitles,
              "maxWord": maxWord, "normalizer": normalizer, "vectorizer": vectorizer,
              "qdBucket": qdBucket, "nFeature": 32, "queryLayers": [32], "candidateLayers": [32]}

    # Generate Data
    tf.random.set_seed(seed)
    shuffled = ratings.shuffle(
        totalNum, seed=seed, reshuffle_each_iteration=False)

    trainSet = shuffled.take(trainNum)
    validSet = shuffled.skip(trainNum).take(validNum)
    trainSet = trainSet.shuffle(totalNum).batch(trainBsz).cache()
    validSet = validSet.batch(validBsz).cache()

    # Train and eval withot timestamp
    params["useTimestamp"] = False
    model = MovielensMultiLayerModel(**params)
    train(trainSet, validSet, model, epochs, lr,
          resultPath="checkpoints/without_stamp_32")

    # Train and eval with timestamp
    params["useTimestamp"] = True
    model = MovielensMultiLayerModel(**params)
    train(trainSet, validSet, model, epochs, lr,
          resultPath="checkpoints/with_stamp_32")

    # Train and eval with timestamp, deep dense layers
    params["queryLayers"] = [128, 64, 32]
    params["candidateLayers"] = [128, 64, 32]
    model = MovielensMultiLayerModel(**params)
    train(trainSet, validSet, model, epochs, lr,
          resultPath="checkpoints/with_stamp_128_64_32")

    # Train and eval with timestamp, wide dense layers
    params["queryLayers"] = [64, 64]
    params["candidateLayers"] = [64, 64]
    model = MovielensMultiLayerModel(**params)
    train(trainSet, validSet, model, epochs, lr,
          resultPath="checkpoints/with_stamp_64_64")
