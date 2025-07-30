import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

data = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)
data.dropna(inplace=True)
data["income"] = data["income"].apply(lambda x: 1 if x == ">50K" else 0)


education_map = {
    "10th": "dropout", "11th": "dropout", "12th": "dropout", "1st-4th": "dropout", "5th-6th": "dropout",
    "7th-8th": "dropout", "9th": "dropout", "Preschool": "dropout", "HS-grad": "high school graduate",
    "Bachelors": "bachelors", "Masters": "masters", "Doctorate": "doctorate", "Prof-school": "professor",
    "Assoc-acdm": "bachelors", "Assoc-voc": "bachelors", "Some-college": "high school graduate"
}
data["education"] = data["education"].map(education_map)


categorical_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
continuous_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

data = pd.get_dummies(data, columns=["education"] + categorical_features, drop_first=True)
scaler = MinMaxScaler()
data[continuous_features] = scaler.fit_transform(data[continuous_features])

X = data.drop("income", axis=1).values
y = data["income"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_
VIM = feature_importances / np.sum(feature_importances)

d = X_train.shape[1]  # 特征数



def add_weighted_random_noise(X, VIM, delta_base=0.1, N=41):
    X_noisy = X.copy()
    n_samples, n_features = X.shape

    for j in range(n_features):
        # --- 步骤1: 定义该特征的噪声范围 -
        # 可根据重要性调整：越重要，扰动越小
        delta = delta_base * (1 - VIM[j])
        noise_candidates = np.linspace(-delta, delta, N)
        similarities = []
        for val in noise_candidates:
            sim = 1 / (1 + abs(val))
            similarities.append(sim)
        similarities = np.array(similarities)
        sorted_idx = np.argsort(similarities)[::-1]
        s_weights = np.ones(N)
        s_weights[sorted_idx[0]] = math.exp(1)

        total_weight = np.sum(s_weights)
        m = 1
        for i in range(1, N):
            idx = sorted_idx[i]
            if similarities[idx] > 0.5:
                s_weights[idx] = math.exp(1)
                m = i + 1
            else:
                break


        total_s = np.sum(s_weights)
        probabilities = s_weights / total_s


        noise = np.random.choice(noise_candidates, size=n_samples, p=probabilities)
        X_noisy[:, j] += noise

    return X_noisy


X_train_noisy = add_weighted_random_noise(X_train, VIM, delta_base=0.1, N=41)


X_train_noisy = X_train_noisy.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

assert np.all(np.isfinite(X_train_noisy)), "X_train_noisy contains invalid values"
assert np.all(np.isfinite(X_test)), "X_test contains invalid values"
assert np.all(np.isfinite(y_train)), "y_train contains invalid values"
assert np.all(np.isfinite(y_test)), "y_test contains invalid values"

model = Sequential([
    Input(shape=(d,)),
    Dense(100, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train_noisy, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))


loss, acc = model.evaluate(X_test, y_test, verbose=0)