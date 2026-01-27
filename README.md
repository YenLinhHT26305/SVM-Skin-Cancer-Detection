🩺 Skin Cancer Detection – Binary & Multi-class Classification
1. Mục tiêu

Xây dựng hệ thống phân loại tổn thương da dựa trên metadata của bộ dữ liệu HAM10000, gồm hai bài toán:

Phân loại nhị phân: Lành tính vs Ác tính

Phân loại đa lớp (7 lớp): Nhận diện loại bệnh cụ thể

Ứng dụng mang tính hỗ trợ quyết định, không thay thế chẩn đoán y khoa.

2. Dữ liệu

Dataset: HAM10000 (Skin Lesion Metadata)

Đặc trưng sử dụng:

age (chuẩn hóa bằng StandardScaler)

sex, localization, dataset, dx_type (one-hot encoding)

3. Phân loại nhị phân (Binary Classification)
Nhãn

Ác tính (1):

Melanoma

Basal Cell Carcinoma

Actinic Keratoses

Lành tính (0):

Melanocytic Nevus

Benign Keratosis

Dermatofibroma

Vascular Lesions

Phương pháp

Thuật toán: Support Vector Machine (SVM) + Optuna + GridSearch, có visualization theo ROC CURVE & AUC, PCA, Confusion Matrix

Đầu ra: Xác suất ác tính (predict_proba)

Ngưỡng phân loại: mặc định 0.5 (có thể điều chỉnh để ưu tiên recall)

4. Phân loại đa lớp (Multi-class – 7 lớp)
Các lớp bệnh

Actinic Keratoses

Basal Cell Carcinoma

Benign Keratosis

Dermatofibroma

Melanocytic Nevus

Melanoma

Vascular Lesions

Phương pháp

Thuật toán: SVM (One-vs-Rest – scikit-learn) +  Optuna + GridSearch, có visualization theo Confusion Matrix

Đầu ra: Xác suất cho từng lớp bệnh

Quy tắc dự đoán: chọn lớp có xác suất cao nhất (argmax)

📌 Ngoài kết quả đa lớp, hệ thống còn đánh giá nguy cơ ác tính bằng cách tổng hợp xác suất của các lớp ác tính (Melanoma, BCC, Actinic Keratoses), phù hợp với mục tiêu y khoa.

5. Kết quả

Binary classification:

Accuracy: 0.8271752085816448
Precision: 0.7665369649805448
Recall: 0.9403341288782816
F1 Score: 0.8445873526259379
AUC : 0.8686

Multi-class classification:

Accuracy : 0.7299051422865701
Precision : 0.6954965844941073
Recall : 0.7299051422865701
F1-score : 0.7050830448827751

F1-score thấp do dữ liệu mất cân bằng

Recall của các lớp hiếm (đặc biệt Melanoma) còn hạn chế

6. Triển khai ( chỉ triển khai trên binary classifier)

Ứng dụng được xây dựng bằng Streamlit, cho phép:

Nhập metadata bệnh nhân

Dự đoán Lành / Ác tính

Hiển thị xác suất cho từng lớp

Chạy ứng dụng:

streamlit run app_svm_binary.py

7. Kết luận

Mô hình SVM cho thấy khả năng hỗ trợ phát hiện nguy cơ ung thư da từ metadata.
Tuy nhiên, do dữ liệu mất cân bằng và thiếu thông tin hình ảnh, hiệu quả phân loại đa lớp còn hạn chế.
Hướng phát triển tiếp theo là kết hợp ảnh da liễu và metadata để cải thiện độ chính xác.
