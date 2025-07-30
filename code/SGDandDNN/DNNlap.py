import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy.random import laplace


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

data = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)
data.dropna(inplace=True)
data["income"] = data["income"].apply(lambda x: 1 if x == ">50K" else 0)


continuous_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

education_map = {
    "10th": "dropout", "11th": "dropout", "12th": "dropout", "1st-4th": "dropout", "5th-6th": "dropout",
    "7th-8th": "dropout", "9th": "dropout", "Preschool": "dropout", "HS-grad": "high school graduate",
    "Bachelors": "bachelors", "Masters": "masters", "Doctorate": "doctorate", "Prof-school": "professor",
    "Assoc-acdm": "bachelors", "Assoc-voc": "bachelors", "Some-college": "high school graduate"
}
data["education"] = data["education"].map(education_map)


data = pd.get_dummies(data, columns=["education"] + categorical_features, drop_first=True)


scaler = MinMaxScaler()
data[continuous_features] = scaler.fit_transform(data[continuous_features])


X = data.drop("income", axis=1).values
y = data["income"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_


epsilon = 1.5
M = np.max(X_train, axis=0)
VIM = feature_importances / np.sum(feature_importances)
alpha = d * VIM * np.min(M) / (np.sum(VIM * M))
eps_per_feature = alpha * epsilon


laplace_noise = np.array([laplace(0, 1/eps, size=X_train.shape[0]) if eps > 0 else np.zeros(X_train.shape[0]) for eps in eps_per_feature]).T
X_train_noisy = X_train + laplace_noise


model = Sequential([
    Input(shape=(d,)),
    Dense(100, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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



model.fit(X_train_noisy, y_train, epochs=10, batch_size=100, validation_data=(X_test, y_test))



loss, acc = model.evaluate(X_test, y_test)

