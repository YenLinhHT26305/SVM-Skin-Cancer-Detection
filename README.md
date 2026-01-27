# 🩺 Skin Cancer Detection – Binary & Multi-class Classification

---

## 1. Mục tiêu
Xây dựng hệ thống **phân loại tổn thương da** dựa trên metadata của bộ dữ liệu **HAM10000**, bao gồm hai bài toán:

- **Phân loại nhị phân (Binary classification)**  
  → Lành tính (Benign) vs Ác tính (Malignant)

- **Phân loại đa lớp (Multi-class – 7 lớp)**  
  → Nhận diện loại bệnh da cụ thể

📌 Ứng dụng mang tính **hỗ trợ quyết định**, **không thay thế chẩn đoán y khoa**.

---

## 2. Dữ liệu
- **Dataset**: HAM10000 – Skin Lesion Metadata

### Đặc trưng sử dụng
- `age` – chuẩn hóa bằng **StandardScaler**
- `sex`, `localization`, `dataset`, `dx_type` – **One-hot encoding**

---

## 3. Phân loại nhị phân (Binary Classification)

### Nhãn
**Ác tính (label = 1):**
- Melanoma  
- Basal Cell Carcinoma  
- Actinic Keratoses  

**Lành tính (label = 0):**
- Melanocytic Nevus  
- Benign Keratosis  
- Dermatofibroma  
- Vascular Lesions  

### Phương pháp
- **Thuật toán**: Support Vector Machine (SVM)
- **Tinh chỉnh siêu tham số**: Optuna + GridSearchCV
- **Visualization**:
  - ROC Curve & AUC
  - PCA (trực quan hóa)
  - Confusion Matrix
- **Đầu ra**: Xác suất ác tính (`predict_proba`)
- **Ngưỡng phân loại**:  
  - Mặc định: `0.5`  
  - Có thể điều chỉnh để **ưu tiên Recall (giảm bỏ sót ác tính)**

---

## 4. Phân loại đa lớp (Multi-class – 7 lớp)

### Các lớp bệnh
1. Actinic Keratoses  
2. Basal Cell Carcinoma  
3. Benign Keratosis  
4. Dermatofibroma  
5. Melanocytic Nevus  
6. Melanoma  
7. Vascular Lesions  

### Phương pháp
- **Thuật toán**: SVM (One-vs-Rest – scikit-learn)
- **Tinh chỉnh siêu tham số**: Optuna + GridSearchCV
- **Visualization**:
  - Confusion Matrix
- **Đầu ra**: Xác suất cho từng lớp bệnh
- **Quy tắc dự đoán**:  
  - Chọn lớp có xác suất cao nhất (**argmax**)

📌 Ngoài kết quả đa lớp, hệ thống còn **đánh giá nguy cơ ác tính** bằng cách tổng hợp xác suất của các lớp:
- Melanoma  
- Basal Cell Carcinoma  
- Actinic Keratoses  

Cách tiếp cận này phù hợp với **mục tiêu y khoa**.

---

## 5. Kết quả

### Binary Classification
- **Accuracy**: 0.8272  
- **Precision**: 0.7665  
- **Recall**: 0.9403  
- **F1-score**: 0.8446  
- **AUC**: 0.8686  

📌 Recall cao cho thấy mô hình **ít bỏ sót các ca ác tính**.

### Multi-class Classification
- **Accuracy**: 0.7299  
- **Precision**: 0.6955  
- **Recall**: 0.7299  
- **F1-score**: 0.7051  

📌 F1-score thấp hơn do:
- Dữ liệu **mất cân bằng**
- Recall của các lớp hiếm (đặc biệt **Melanoma**) còn hạn chế

---

## 6. Triển khai
*(Chỉ triển khai giao diện cho bài toán Binary Classification)*

Ứng dụng được xây dựng bằng **Streamlit**, cho phép:
- Nhập metadata bệnh nhân
- Dự đoán **Lành / Ác tính**
- Hiển thị **xác suất ác tính**

Chạy ứng dụng:
```bash
streamlit run app_svm_binary.py
