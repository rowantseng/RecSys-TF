import tensorflow as tf


def lookupEmbSequentialModel(vocabs, nFeature=32):
    return tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=vocabs, mask_token=None),
        tf.keras.layers.Embedding(len(vocabs) + 1, nFeature),
    ])


def qdEmbSequentialModel(buckets, nFeature=32):
    return tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Discretization(buckets),
        tf.keras.layers.Embedding(len(buckets) + 1, nFeature),
    ])


def buildNormalizer(x):
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(x)
    return normalizer


def buildVectorizer(x, maxWord=10000):
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=maxWord)
    vectorizer.adapt(x)
    return vectorizer
