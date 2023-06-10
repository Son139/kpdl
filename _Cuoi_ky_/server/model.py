import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle


class Model:
    def __init__(self):
        self.data = pd.read_csv("purchase_history.csv")
        self.model = None
        self.scaler = None

    def preprocess_data(self):
        gender_encoded = pd.get_dummies(self.data["Gender"], drop_first=True, dtype=int)
        self.data = pd.concat([self.data, gender_encoded], axis=1)
        x = self.data[["Male", "Age", "Salary", "Price"]].to_numpy()
        y = self.data["Purchased"].to_numpy()
        return x, y

    def train_model(self):
        x, y = self.preprocess_data()

        # Huấn luyện mô hình
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=40
        )
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)

        k = 1
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        with open("./models/knn_model.pickle", "wb") as f:
            pickle.dump(self.model, f)

        with open("./models/scaler.pickle", "wb") as f:
            pickle.dump(self.model, f)

        return accuracy

    def predict(self, Male, Age, Salary, Price):
        with open("./models/knn_model.pickle", "rb") as f:
            self.model = pickle.load(f)

        if self.model is None:
            self.train_model()
            print("voday")

        row_value = [Male, Age, Salary, Price]
        x_new = np.array(row_value).reshape(1, -1)
        y_new_pred = self.model.predict(x_new)
        predictions = str(y_new_pred[0])
        return predictions

    def upload_csv(self, filename):
        # with open('./models/scaler.pickle', 'rb') as f:
        #     self.scaler = pickle.load(f)

        if self.scaler is None:
            self.train_model()
            print("voday")

        new_df = pd.read_csv(filename)
        gender_encoded_new = pd.get_dummies(
            new_df["Gender"], drop_first=True, dtype=int
        )

        df_new_2 = pd.concat([new_df, gender_encoded_new], axis=1)
        x_new = df_new_2[["Male", "Age", "Salary", "Price"]].to_numpy()

        x_new_scale2 = self.scaler.fit_transform(x_new)

        y_new_pred = self.model.predict(x_new_scale2)

        df_new_2["will_purchase"] = y_new_pred

        return df_new_2

    def train(self, filename, k, test_size, random_state):
        self.data = pd.read_csv(filename)

        x, y = self.preprocess_data()

        # Huấn luyện mô hình
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)

        # k = 1
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        with open("./models/knn_model.pickle", "wb") as f:
            pickle.dump(self.model, f)

        with open("./models/scaler.pickle", "wb") as f:
            pickle.dump(self.scaler, f)

        return accuracy