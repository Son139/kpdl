import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

class Model:
    def __init__(self):
        self.data = pd.read_csv('purchase_history.csv')
        self.model = None

    def preprocess_data(self, X):
        """Tiền xử lý dữ liệu"""
        X['category'] = X['category'].astype('category')
        X = pd.get_dummies(X, columns=['category'])
        return X

    def train_model(self):
        """Huấn luyện mô hình KNN"""
        X = self.data[['price', 'time', 'category']]
        y = self.data['buy']
        X = self.preprocess_data(X)
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X, y)

        # Lưu mô hình vào file
        with open('model.pkl', 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, price, time, category):
        """Dự đoán hành vi mua hàng"""
        if self.model is None:
            # Nếu chưa có mô hình thì huấn luyện mô hình trước khi dự đoán
            self.train_model()

        category_col = 'category_' + category
        X_pred = [[price, time] + [0] * (len(self.model.columns) - 2)]
        X_pred[0][self.model.columns.get_loc('price')] = price
        X_pred[0][self.model.columns.get_loc('time')] = time
        X_pred[0][self.model.columns.get_loc(category_col)] = 1
        prediction = self.model.predict(X_pred)

        # Trả về kết quả dự đoán
        if prediction[0] == 1:
            result = 'Khách hàng có thể mua sản phẩm'
        else:
            result = 'Khách hàng không mua sản phẩm'

        return result