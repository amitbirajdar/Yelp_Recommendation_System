from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from pyspark import SparkConf, SparkContext
import csv, time, json, sys
import numpy as np
import pandas as pd
import xgboost as xgb


def calculate_pearson_correlation(rating):

    # rating = [user_id, business_id, rating]
    test_user = rating[0]
    test_business = rating[1]
    coefficients = []

    # Tackling cold start problem
    # new user
    if (test_user not in user_dict):
        # if user and business both are new
        if (test_business not in business_dict):
            return(test_user, test_business, 3.5)
        else:
            return(test_user, test_business, business_avg_rating_dict[test_business])

    # new business
    if (test_business not in business_dict):
        # if business and user both are new
        if (test_user not in user_dict):
            return(test_user, test_business, 3.5)
        else:
            return(test_user, test_business, user_avg_rating_dict[test_user])


    businesses_rated_by_user = user_dict[test_user]

    if len(businesses_rated_by_user) > 0:
        for rated_business in businesses_rated_by_user:

            # print(test_user, rated_business)
            rating_for_rated_business = user_business_rating_dict[(test_user, rated_business)]

            # finding all users who have rated both Bi and Bj
            rated_Bi = set(business_dict[test_business])
            rated_Bj = set(business_dict[rated_business])
            neighboring_users = rated_Bi.intersection(rated_Bj)

            # if less than 2 neighboring users found -> business rating = avg rating for that business
            if len(neighboring_users) < 50:
                continue

            # If number of neighbors > 2 ->
            ratings_Bi, ratings_Bj = [], []

            for user in neighboring_users:

                # ratings for business Bi
                ratings_Bi.append(user_business_rating_dict[(user, test_business)])

                # ratings for business Bj
                ratings_Bj.append(user_business_rating_dict[(user, rated_business)])

            # Calculate pearson correlation

            multiplication = []
            power_Bi, power_Bj = [], []

            for i in range(len(ratings_Bi)):
                multiplication.append(ratings_Bi[i] * ratings_Bj[i])
                power_Bi.append(ratings_Bi[i]**2)
                power_Bj.append(ratings_Bj[i]**2)
            
            numerator = sum(multiplication)
            den_i = sum(power_Bi)**0.5
            den_j = sum(power_Bj)**0.5
            
            denominator = den_i * den_j

            if numerator == 0 or denominator == 0:
                coefficients.append([0.0, 0.0, 0.0])
                continue
            
            coeff = numerator / denominator

            coefficients.append([coeff, coeff*rating_for_rated_business, abs(coeff)])
    else:
        return(test_user, test_business, business_avg_rating_dict[test_business])

    # if no coefficients
    if len(coefficients) == 0:
        return(test_user, test_business, business_avg_rating_dict[test_business])

    # Selecting top N neighbors
    if len(coefficients) > N:
        selected_neighbors = np.array(sorted(coefficients, key=lambda x: -x[0])[:N])
    else:
        selected_neighbors = np.array(sorted(coefficients, key=lambda x: -x[0]))
    
    prediction = predict(selected_neighbors, test_user, test_business)

    # Prediction
    return(prediction)
    
###############################################################################################################################

def predict(selected_neighbors, test_user, test_business):
    
    selected_neighbors_summation = selected_neighbors.sum(axis=0)

    p_numerator = selected_neighbors_summation[1]
    p_denominator = selected_neighbors_summation[2]

    if p_numerator == 0 or p_denominator == 0:
        return(test_user, test_business, 0)

    prediction = p_numerator / p_denominator
    return(test_user, test_business, prediction)

###############################################################################################################################

def write_results(df_res, output_file):

    df_res.to_csv(output_file, index=False)


###############################################################################################################################

def create_train_df(train_df, user_data, business_data):
    
    user = train_df['user_id'].tolist()
    business = train_df['business_id'].tolist()
    star = train_df['stars'].tolist()
    
    user_rows = []
    
    for i in range(len(user)):
    
        vals1,vals2 = [], []
    
        if user[i] in user_data:
            for k in user_data[user[i]]:
                vals1.append(k)
            #vals1 = [k for k in user_data[user[i]]]

            if business[i] in business_data:
                for k in business_data[business[i]]:
                    vals2.append(k)
                # vals2 = [l for l in business_data[business[i]]]

                if len(vals1) != 0 or len(vals2) != 0:
                    user_rows.append(list( [user[i]] + [business[i]] + vals1 + vals2 + [float(star[i])] ) )

    return user_rows

###############################################################################################################################

def create_test_df(test_df, user_data, business_data):

    user = test_df['user_id'].tolist()
    business = test_df['business_id'].tolist()
    
    user_rows = []
    
    for i in range(len(user)):
    
        vals1, vals2 = [], []
        if user[i] in user_data:
            for k in user_data[user[i]]:
                vals1.append(k)

        if business[i] in business_data:
            for k in business_data[business[i]]:
                vals2.append(k)

        if len(vals1) != 0 or len(vals2) != 0:
            user_rows.append(list( [user[i]] + [business[i]] + vals1 + vals2 ) )
    
    return user_rows
        

###############################################################################################################################

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sc.setLogLevel("WARN")
args = sys.argv

start_time = time.time()

N = 50

train_data_file = sc.textFile(args[1]+"/yelp_train.csv")

val_data_file = sc.textFile(args[2])

train_data = train_data_file.filter(lambda header: not header.startswith("user_id")).map(lambda line: line.split(","))

val_data = val_data_file.filter(lambda header: not header.startswith("user_id")).map(lambda line: line.split(","))

user_dict = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

business_dict = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()

user_business_rating_dict = train_data.map(lambda x: ((x[0], x[1]), float(x[2]))).collectAsMap()
    
user_avg_rating_dict = train_data.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

business_avg_rating_dict = train_data.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

predictions_item_based = val_data.map(calculate_pearson_correlation).collect()

train_df = pd.read_csv(args[1]+"/yelp_train.csv")
test_df = pd.read_csv(args[2])

photo1 = sc.textFile(args[1]+"/photo.json").map(lambda file: json.loads(file)).collect()
photo2 = json.dumps(photo1)
photo3 = pd.read_json(photo2)
photos = dict(photo3.business_id.value_counts())
photos = pd.DataFrame.from_dict(photos, orient="index")
photos = photos.reset_index()
photos = photos.rename(columns={"index":"business_id", 0:"photo_count"})

tip1 = sc.textFile(args[1]+"/tip.json").map(lambda file: json.loads(file)).collect()
tip2 = json.dumps(tip1)
tip3 = pd.read_json(tip2)
tip = dict(tip3.user_id.value_counts())
tip = pd.DataFrame.from_dict(tip, orient="index")
tip = tip.reset_index()
tip = tip.rename(columns={"index":"user_id", 0:"tip_count"})


user_json_file = sc.textFile(args[1]+"/user.json").map(lambda file: json.loads(file))
user_data = user_json_file.map(lambda x : ( ( x['user_id'], ( x['review_count'], x['average_stars'] ) ) ) ).collectAsMap()
business_json_file = sc.textFile(args[1]+"/business.json").map(lambda file: json.loads(file))
business_data = business_json_file.map(lambda x : ( ( x['business_id'], ( int(x['review_count']), x['stars'], x['latitude'], x['longitude'], int(x['is_open']) ) ) ) ).collectAsMap()

user_rows = create_train_df(train_df,user_data,business_data)
df_train = pd.DataFrame(user_rows, columns=['user_id','business_id','user_review_count','average_stars','business_review_count','business_stars', 'latitude', 'longitude', 'is_open', 'stars'])
df_train = df_train.merge(photos, on="business_id", how="left")
df_train = df_train.merge(tip, on="user_id", how="left")

test_rows = create_test_df(test_df,user_data,business_data)
df_test = pd.DataFrame(test_rows, columns=['user_id','business_id','user_review_count','average_stars','business_review_count','business_stars', 'latitude', 'longitude', 'is_open'])
df_test = df_test.merge(photos, on="business_id", how="left")
df_test = df_test.merge(tip, on="user_id", how="left")

df_train.rename(columns={"photo_count":"no_of_photos"}, inplace=True)
df_test.rename(columns={"photo_count":"no_of_photos"}, inplace=True)

df_train.no_of_photos.fillna(0, inplace=True)
df_test.no_of_photos.fillna(0, inplace=True)

df_train.no_of_photos = df_train.no_of_photos.astype("int")
df_test.no_of_photos = df_test.no_of_photos.astype("int")

df_train.tip_count.fillna(0, inplace=True)
df_test.tip_count.fillna(0, inplace=True)

df_train.tip_count = df_train.tip_count.astype("int")
df_test.tip_count = df_test.tip_count.astype("int")

X_train, y_train = df_train.drop(['user_id','business_id','stars'], axis=1), df_train["stars"]

X_val = df_test.drop(['user_id','business_id'], axis=1)

model = xgb.XGBRegressor(random_state = 7, learning_rate=0.5, n_estimators=250)
model.fit(X_train,y_train)
    
predictions_model_based = model.predict(X_val)

p = []
for prediction in predictions_item_based:
    p.append(prediction[2])

p1 = np.array(p, dtype=float)
p2 = np.array(predictions_model_based, dtype=float)

alpha = 0.95

final_prediction = alpha*p2 + (1-alpha)*p1


df_res = df_test[["user_id", "business_id"]]
df_res["prediction"] = final_prediction

write_results(df_res, args[3])

end_time = time.time()
print(f"Execution Time = {end_time - start_time}")