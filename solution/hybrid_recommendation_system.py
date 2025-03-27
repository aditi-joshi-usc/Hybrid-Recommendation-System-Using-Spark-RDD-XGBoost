from pyspark import SparkContext
from xgboost import XGBRegressor
import sys
import numpy as np
import time
import json


def load_training_data(sc, data_path):
    train_rdd = (sc.textFile(f"{data_path}/yelp_train.csv")
                 .zipWithIndex()
                 .filter(lambda x: x[1] > 0)  # Skip header
                 .map(lambda x: x[0].split(","))
                 .map(lambda x: (x[0], x[1], x[2])))  # (user_id, business_id, rating)
    return train_rdd

def load_photos_features(sc, data_path):
    def cnt_label(labels):
        label_count = {}
        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1
        return label_count

    photo_rdd = sc.textFile(f"{data_path}/photo.json") \
                  .map(json.loads) \
                  .map(lambda x: (x['business_id'], [x["label"]])) \
                  .reduceByKey(lambda x, y: x + y)
    
    photos = {}
    for business_id, labels in photo_rdd.collect():
        rcd = cnt_label(labels)
        rcd["photo_sum"] = sum(rcd.values())
        photos[business_id] = rcd  # Keep business_id and photo features as a dictionary
    return photos


def load_review_features(sc, data_path):
    review_rdd = sc.textFile(f"{data_path}/review_train.json").map(json.loads)
    review_features = review_rdd.map(lambda x: (x['business_id'], (x['useful'], x['funny'], x['cool']))) \
                                .groupByKey().mapValues(list).collectAsMap()
    business_review_features = {}
    for business_id, features in review_features.items():
        num_reviews = len(features)
        avg_features = tuple(sum(f[i] for f in features) / num_reviews for i in range(3))
        business_review_features[business_id] = avg_features
    return business_review_features

def load_tips_features(sc, data_path):
    tips_rdd = (sc.textFile(f"{data_path}/tip.json")
                .map(json.loads)
                .map(lambda x: (x['business_id'], (x['likes'], 1)))
                .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
                .map(lambda x: (x[0], (x[1][0], x[1][1])))  # (business_id, (likes_sum, likes_count))
                .collectAsMap())
    return tips_rdd


def load_user_features(sc, data_path):
    user_rdd = sc.textFile(f"{data_path}/user.json").map(json.loads)
    return user_rdd.map(lambda x: (x['user_id'], (x['average_stars'], x['review_count'], x['fans']))) \
                   .collectAsMap()


def load_business_features(sc, data_path):
    business_rdd = sc.textFile(f"{data_path}/business.json").map(json.loads)
    return business_rdd.map(lambda x: (x['business_id'], (x['stars'], x['review_count']))) \
                       .collectAsMap()


# def prepare_mappings(train_data):
#     user_business_map = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
#     business_user_map = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()

#     def average_ratings(data, index):
#         return data.map(lambda x: (x[index], float(x[2]))).groupByKey() \
#                    .mapValues(lambda ratings: sum(ratings) / len(ratings)).collectAsMap()

#     user_avg_ratings = average_ratings(train_data, 0)
#     business_avg_ratings = average_ratings(train_data, 1)

#     business_user_ratings = train_data.map(lambda x: (x[1], (x[0], float(x[2])))) \
#                                       .groupByKey().mapValues(dict).collectAsMap()

#     return user_business_map, business_user_map, user_avg_ratings, business_avg_ratings, business_user_ratings
def prepare_mappings(train_data):
    user_business_map = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    business_user_map = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()

    def average_ratings(data, index):
        return data.map(lambda x: (x[index], float(x[2]))).groupByKey() \
                   .mapValues(lambda ratings: sum(ratings) / len(ratings)).collectAsMap()

    user_avg_ratings = average_ratings(train_data, 0)
    business_avg_ratings = average_ratings(train_data, 1)

    # Global Average Rating
    global_avg_rating = train_data.map(lambda x: float(x[2])).mean()

    # User and Business Biases
    user_bias = {user: user_avg_ratings[user] - global_avg_rating for user in user_avg_ratings}
    business_bias = {business: business_avg_ratings[business] - global_avg_rating for business in business_avg_ratings}

    business_user_ratings = train_data.map(lambda x: (x[1], (x[0], float(x[2])))) \
                                      .groupByKey().mapValues(dict).collectAsMap()

    return user_business_map, business_user_map, user_avg_ratings, business_avg_ratings, business_user_ratings, global_avg_rating, user_bias, business_bias


def compute_similarity(business_id, other_business, business_user_map, business_avg_ratings, business_user_ratings, similarity_cache):
    key = tuple(sorted((business_id, other_business)))
    if key in similarity_cache:
        return similarity_cache[key]

    common_users = business_user_map.get(business_id, set()) & business_user_map.get(other_business, set())
    if len(common_users) < 2:
        avg_rating_diff = abs(business_avg_ratings.get(business_id, 3.5) - business_avg_ratings.get(other_business, 3.5))
        similarity = (5.0 - avg_rating_diff) / 5
    else:
        ratings_a = [business_user_ratings[business_id][user] for user in common_users]
        ratings_b = [business_user_ratings[other_business][user] for user in common_users]
        avg_a = sum(ratings_a) / len(ratings_a)
        avg_b = sum(ratings_b) / len(ratings_b)
        numerator = sum((ra - avg_a) * (rb - avg_b) for ra, rb in zip(ratings_a, ratings_b))
        denominator = (sum((ra - avg_a) ** 2 for ra in ratings_a) ** 0.5) * \
                      (sum((rb - avg_b) ** 2 for rb in ratings_b) ** 0.5)
        similarity = numerator / denominator if denominator != 0 else 0

    similarity_cache[key] = similarity
    return similarity


# def item_based_prediction(user_id, business_id, mappings, similarity_cache):
#     user_business_map, business_user_map, user_avg_ratings, business_avg_ratings, business_user_ratings = mappings

#     if user_id not in user_business_map:
#         return 3.5
#     if business_id not in business_user_map:
#         return user_avg_ratings.get(user_id, 3.5)

#     similarities = [
#         (compute_similarity(business_id, other_business, business_user_map, business_avg_ratings, business_user_ratings, similarity_cache),
#          business_user_ratings[other_business][user_id])
#         for other_business in user_business_map[user_id]
#     ]

#     # Select top 15 most similar businesses
#     top_similarities = sorted(similarities, key=lambda x: -x[0])[:10]
#     numerator = sum(sim * rating for sim, rating in top_similarities)
#     denominator = sum(abs(sim) for sim, _ in top_similarities)
#     return numerator / denominator if denominator != 0 else 3.5
def item_based_prediction(user_id, business_id, mappings, similarity_cache):
    (user_business_map, business_user_map, user_avg_ratings, business_avg_ratings,
     business_user_ratings, global_avg_rating, user_bias, business_bias) = mappings

    if user_id not in user_business_map:
        return global_avg_rating + user_bias.get(user_id, 0) + business_bias.get(business_id, 0)
    if business_id not in business_user_map:
        return global_avg_rating + user_bias.get(user_id, 0) + business_bias.get(business_id, 0)

    similarities = [
        (compute_similarity(business_id, other_business, business_user_map, business_avg_ratings, business_user_ratings, similarity_cache),
         business_user_ratings[other_business][user_id])
        for other_business in user_business_map[user_id]
    ]

    # Select top 15 most similar businesses
    top_similarities = sorted(similarities, key=lambda x: -x[0])[:10]
    numerator = sum(sim * rating for sim, rating in top_similarities)
    denominator = sum(abs(sim) for sim, _ in top_similarities)
    cf_prediction = numerator / denominator if denominator != 0 else global_avg_rating

    # Adjust CF prediction with baseline bias
    return cf_prediction + user_bias.get(user_id, 0) + business_bias.get(business_id, 0)


def collaborative_filtering_predictions(sc, val_data_path, mappings):
    val_data = sc.textFile(val_data_path).zipWithIndex() \
                .filter(lambda x: x[1] > 0) \
                .map(lambda x: x[0].split(",")) \
                .map(lambda x: (x[0], x[1]))  # Select user_id and business_id columns

    similarity_cache = {}
    predictions = []
    for user_id, business_id in val_data.collect():
        pred = item_based_prediction(user_id, business_id, mappings, similarity_cache)
        predictions.append((user_id, business_id, pred))
    return predictions


# def prepare_features(train_data, user_features, business_features, review_features):
#     X_train, Y_train = [], []
#     for user_id, business_id, rating in train_data.collect():
#         user_f = user_features.get(user_id, (0.0, 0.0, 0.0))
#         business_f = business_features.get(business_id, (0.0, 0.0))
#         review_f = review_features.get(business_id, (0.0, 0.0, 0.0))
#         X_train.append(list(review_f) + list(user_f) + list(business_f))
#         Y_train.append(float(rating))
#     return np.array(X_train, dtype='float32'), np.array(Y_train, dtype='float32')

def prepare_features(train_data, user_features, business_features, review_features, tips_features, photos_features):
    X_train, Y_train = [], []
    for user_id, business_id, rating in train_data.collect():
        user_f = user_features.get(user_id, (0.0, 0.0, 0.0))
        business_f = business_features.get(business_id, (0.0, 0.0))
        review_f = review_features.get(business_id, (0.0, 0.0, 0.0))
        tips_f = tips_features.get(business_id, (0.0, 0.0))
        photo_f = photos_features.get(business_id, {"photo_sum": 0}).get("photo_sum", 0)  # Use only photo_sum for simplicity
        X_train.append(list(review_f) + list(user_f) + list(business_f) + list(tips_f) + [photo_f])
        Y_train.append(float(rating))
    return np.array(X_train, dtype='float32'), np.array(Y_train, dtype='float32')



# def model_based_predictions(sc, val_data_path, user_features, business_features, review_features, model):
#     val_data = sc.textFile(val_data_path).zipWithIndex() \
#                 .filter(lambda x: x[1] > 0) \
#                 .map(lambda x: x[0].split(",")) \
#                 .map(lambda x: (x[0], x[1]))  # Select user_id and business_id columns

#     X_val = []
#     user_business_pairs = []
#     for user_id, business_id in val_data.collect():
#         user_f = user_features.get(user_id, (0.0, 0.0, 0.0))
#         business_f = business_features.get(business_id, (0.0, 0.0))
#         review_f = review_features.get(business_id, (0.0, 0.0, 0.0))
#         X_val.append(list(review_f) + list(user_f) + list(business_f))
#         user_business_pairs.append((user_id, business_id))
#     X_val = np.array(X_val, dtype='float32')
#     Y_pred = model.predict(X_val)
#     return list(zip(user_business_pairs, Y_pred))
def model_based_predictions(sc, val_data_path, user_features, business_features, review_features, tips_features, photos_features, model):
    val_data = sc.textFile(val_data_path).zipWithIndex() \
                .filter(lambda x: x[1] > 0) \
                .map(lambda x: x[0].split(",")) \
                .map(lambda x: (x[0], x[1]))  # Select user_id and business_id columns

    X_val = []
    user_business_pairs = []
    for user_id, business_id in val_data.collect():
        user_f = user_features.get(user_id, (0.0, 0.0, 0.0))
        business_f = business_features.get(business_id, (0.0, 0.0))
        review_f = review_features.get(business_id, (0.0, 0.0, 0.0))
        tips_f = tips_features.get(business_id, (0.0, 0.0))
        photo_f = photos_features.get(business_id, {"photo_sum": 0}).get("photo_sum", 0)  # Use only photo_sum
        X_val.append(list(review_f) + list(user_f) + list(business_f) + list(tips_f) + [photo_f])
        user_business_pairs.append((user_id, business_id))
    X_val = np.array(X_val, dtype='float32')
    Y_pred = model.predict(X_val)
    return list(zip(user_business_pairs, Y_pred))



def train_model(X_train, Y_train):
    params = {
        'lambda': 9.93,
        'alpha': 0.28,
        'colsample_bytree': 0.5,
        'subsample': 0.7,
        'learning_rate': 0.01,
        'max_depth': 15,
        'random_state': 2050,
        'min_child_weight': 80,
        'n_estimators': 600,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, Y_train)
    return model


def blend_predictions(cf_preds, model_preds, blend_factor=0.05):
    blended = {}
    model_pred_dict = {pair: pred for pair, pred in model_preds}
    for user_id, business_id, cf_pred in cf_preds:
        model_pred = model_pred_dict.get((user_id, business_id), 3.5)
        final_pred = blend_factor * cf_pred + (1 - blend_factor) * model_pred
        blended[(user_id, business_id)] = final_pred
    return blended


def write_predictions(predictions, output_path):
    with open(output_path, 'w') as f:
        f.write("user_id,business_id,prediction\n")
        for (user_id, business_id), prediction in predictions.items():
            f.write(f"{user_id},{business_id},{prediction}\n")
    return predictions


def evaluate_predictions(predictions, val_data_path, sc):
    # Load validation data
    val_data = sc.textFile(val_data_path).zipWithIndex() \
                .filter(lambda x: x[1] > 0) \
                .map(lambda x: x[0].split(",")) \
                .map(lambda x: (x[0], x[1], float(x[2])))  # Include actual ratings

    errors = []
    for user_id, business_id, actual_rating in val_data.collect():
        predicted_rating = predictions.get((user_id, business_id), 3.5)
        errors.append(predicted_rating - actual_rating)

    # Compute RMSE
    rmse = np.sqrt(np.mean([e ** 2 for e in errors]))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Compute error distribution
    error_bins = {}
    for error in errors:
        bin_key = round(error, 1)  # Group errors to the nearest 0.1
        if bin_key not in error_bins:
            error_bins[bin_key] = 0
        error_bins[bin_key] += 1

    # Print error distribution
    print("\nError Distribution:")
    for bin_key in sorted(error_bins.keys()):
        print(f"Error: {bin_key}, Count: {error_bins[bin_key]}")


def main():
    _, data_path, val_data_path, output_path = sys.argv
    start_time = time.time()

    sc = SparkContext(appName="HybridRecommendationSystem")
    
    # Load data
    train_data = load_training_data(sc, data_path)
    user_features = load_user_features(sc, data_path)
    business_features = load_business_features(sc, data_path)
    review_features = load_review_features(sc, data_path)
    tips_features = load_tips_features(sc, data_path)  # Load tips data
    photos_features = load_photos_features(sc, data_path)  # Load photos data
    
    mappings = prepare_mappings(train_data)
    
    cf_preds = collaborative_filtering_predictions(sc, val_data_path, mappings)
    
    X_train, Y_train = prepare_features(train_data, user_features, business_features, review_features, tips_features, photos_features)
    model = train_model(X_train, Y_train)
    
    model_preds = model_based_predictions(sc, val_data_path, user_features, business_features, review_features, tips_features, photos_features, model)
    
    blended_preds = blend_predictions(cf_preds, model_preds, blend_factor=0.05)
    
    blended_preds = write_predictions(blended_preds, output_path)
    
    # Evaluate predictions
    evaluate_predictions(blended_preds, val_data_path, sc)
    
    end_time = time.time()
    print(f"Duration: {end_time - start_time}")


if __name__ == '__main__':

    main()