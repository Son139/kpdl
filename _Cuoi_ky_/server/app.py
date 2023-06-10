from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from model import Model

# Tạo ứng dụng web Flask
app = Flask(__name__)
# CORS(app, supports_credentials=True)
CORS(app, supports_credentials=True, origins='http://localhost:3000')

app.config['CORS_HEADERS'] = 'Content-Type'
# Định nghĩa trang chủ
@app.route('/')
def home():
    return "Welcome to API Server!!!"

# Định nghĩa trang lấy giá trị accuracy
@app.route('/accuracy', methods=['GET'])
def accuracy():
    model = Model()
    acc = model.train_model()
    return jsonify(acc)

# Định nghĩa predict
@app.route('/predict', methods=['POST'])
def prediction():
    # Lấy thông tin từ form
    male = float(request.get_json()['male'])
    age = float(request.get_json()['age'])
    salary = float(request.get_json()['salary'])
    price = float(request.get_json()['price'])
    model = Model()
    result = model.predict(male, age, salary, price )
    return jsonify(result)

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    # Lấy file từ form
    file = request.files["file"]
    model = Model()
    result = model.upload_csv(file)
    return result.to_json()

# train
@app.route("/train", methods=["POST"])
def train():
    # Nhận các tham số truyền vào từ request
    file = request.files["file"]
    model = Model()
    k = float(request.get_json["k"])
    test_size = float(request.get_json["test_size"])
    random_state = int(request.get_json["random_state"])
    accuracy = model.train(file, k, test_size, random_state)
    return jsonify(accuracy)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
