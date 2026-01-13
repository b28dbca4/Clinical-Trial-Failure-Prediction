# Dự Đoán Thất Bại Thử Nghiệm Lâm Sàng Ung Thư

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)

Dự án này áp dụng các kỹ thuật Khoa học Dữ liệu và Học máy để phân tích rủi ro và xây dựng mô hình dự đoán khả năng thất bại của các thử nghiệm lâm sàng ung thư, nhằm hỗ trợ ra quyết định chiến lược trong nghiên cứu phát triển dược phẩm.

---

## Mục Lục

1. [Giới Thiệu](#1-giới-thiệu)
2. [Dataset](#2-dataset)
3. [Phương Pháp Nghiên Cứu](#3-phương-pháp-nghiên-cứu)
4. [Cài Đặt và Thiết Lập](#4-cài-đặt-và-thiết-lập)
5. [Hướng Dẫn Sử Dụng](#5-hướng-dẫn-sử-dụng)
6. [Kết Quả](#6-kết-quả)
7. [Cấu Trúc Dự Án](#7-cấu-trúc-dự-án)
8. [Thách Thức và Giải Pháp](#8-thách-thức-và-giải-pháp)
9. [Hướng Phát Triển](#9-hướng-phát-triển)
10. [Tác Giả](#10-tác-giả)
11. [Giấy Phép](#11-giấy-phép)

---

## 1. Giới Thiệu

### 1.1. Mô Tả Bài Toán

Thử nghiệm lâm sàng là giai đoạn quan trọng và tốn kém nhất trong quy trình phát triển thuốc. Đặc biệt trong lĩnh vực ung thư, tỷ lệ thất bại của các thử nghiệm lâm sàng dao động từ 70% đến 90%, với chi phí trung bình từ 50 đến 100 triệu USD cho mỗi thử nghiệm. Bài toán đặt ra là làm thế nào để tận dụng dữ liệu lịch sử để xác định sớm các thử nghiệm có nguy cơ thất bại cao, từ đó đưa ra quyết định Go/No-Go kịp thời.

Dự án này tập trung vào bài toán phân loại nhị phân (Binary Classification) để dự đoán kết quả thử nghiệm lâm sàng ung thư:
- Lớp 0 (Thất bại): Thử nghiệm bị TERMINATED hoặc WITHDRAWN
- Lớp 1 (Thành công): Thử nghiệm COMPLETED

### 1.2. Động Lực và Ứng Dụng Thực Tế

Việc dự đoán chính xác kết cục thử nghiệm mang lại giá trị to lớn trong thực tiễn:

- Tối ưu hóa nguồn lực: Phát hiện sớm các thử nghiệm có nguy cơ thất bại cao giúp tiết kiệm đáng kể chi phí tài chính và nhân lực, cho phép tái phân bổ nguồn lực cho các dự án có triển vọng hơn.

- An toàn bệnh nhân: Giảm thiểu rủi ro cho người tham gia bằng cách tránh kéo dài các thử nghiệm không khả thi.

- Hỗ trợ ra quyết định: Cung cấp công cụ sàng lọc ban đầu (screening tool) cho các công ty dược phẩm và nhà đầu tư trong việc đánh giá các thử nghiệm mới.

- Cải thiện thiết kế nghiên cứu: Xác định các yếu tố then chốt ảnh hưởng đến thành công để tối ưu hóa thiết kế thử nghiệm ngay từ giai đoạn đầu.

### 1.3. Mục Tiêu Cụ Thể

Dự án được thực hiện với các mục tiêu cụ thể sau:

1. Thu thập dữ liệu từ cơ sở dữ liệu ClinicalTrials.gov thông qua API công khai, tập trung vào các thử nghiệm lâm sàng ung thư Phase 2 và Phase 3.

2. Tiền xử lý và làm sạch dữ liệu, xử lý giá trị thiếu và chuẩn hóa các đặc trưng theo quy trình khoa học.

3. Phân tích khám phá dữ liệu (EDA) để trả lời 5 câu hỏi nghiên cứu về các yếu tố ảnh hưởng đến kết quả thử nghiệm.

4. Xây dựng và đánh giá các mô hình học máy (Logistic Regression, Random Forest, XGBoost, LightGBM) để dự đoán thất bại thử nghiệm.

5. Phân tích feature importance và đưa ra các khuyến nghị dựa trên kết quả mô hình.

---

## 2. Dataset

### 2.1. Nguồn Dữ Liệu

Dữ liệu được thu thập từ ClinicalTrials.gov - cơ sở dữ liệu thử nghiệm lâm sàng lớn nhất thế giới do Viện Y tế Quốc gia Hoa Kỳ (NIH) quản lý.

- Nền tảng: ClinicalTrials.gov API v2
- URL: https://clinicaltrials.gov/api/v2/studies
- Tác giả: U.S. National Library of Medicine (NLM)
- Phạm vi: Thử nghiệm can thiệp (INTERVENTIONAL), Phase 2-3, từ khóa "cancer"
- Thời gian: Các thử nghiệm từ năm 2000 đến nay
- Giấy phép: Dữ liệu công khai theo chính sách NIH, được phép sử dụng cho mục đích nghiên cứu và giáo dục

### 2.2. Mô Tả Các Features

Bộ dữ liệu bao gồm 23 cột nguyên bản và được mở rộng thành 34 đặc trưng sau quá trình feature engineering. Các nhóm đặc trưng chính bao gồm:

Nhóm Định danh và Mô tả:
- `nct_id`: Mã định danh duy nhất của thử nghiệm trên ClinicalTrials.gov
- `brief_title`: Tiêu đề ngắn gọn mô tả nội dung thử nghiệm

Nhóm Thời gian:
- `start_date`: Ngày bắt đầu thử nghiệm
- `completion_date`: Ngày hoàn thành dự kiến hoặc thực tế
- `duration_days`: Thời gian thực hiện thử nghiệm (tính bằng ngày)

Nhóm Thiết kế Nghiên cứu:
- `phases`: Giai đoạn thử nghiệm (PHASE2, PHASE3, PHASE2|PHASE3)
- `allocation`: Phương pháp phân bổ (RANDOMIZED, NON_RANDOMIZED, UNKNOWN)
- `intervention_model`: Mô hình can thiệp (PARALLEL, SINGLE_GROUP, CROSSOVER, FACTORIAL, SEQUENTIAL)
- `masking`: Phương pháp làm mù (NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE, UNKNOWN)
- `primary_purpose`: Mục đích chính (TREATMENT, DIAGNOSTIC, PREVENTION, SUPPORTIVE_CARE, v.v.)

Nhóm Quy mô:
- `enrollment_count`: Số lượng người tham gia
- `num_arms`: Số nhánh nghiên cứu
- `enrollment_per_arm`: Số người tham gia trung bình mỗi nhánh

Nhóm Nhà tài trợ:
- `lead_sponsor_name`: Tên nhà tài trợ chính
- `lead_sponsor_class`: Phân loại nhà tài trợ (INDUSTRY, NIH, OTHER, NETWORK, FED, INDIV, OTHER_GOV, UNKNOWN)
- `has_dmc`: Có Ủy ban Giám sát An toàn Dữ liệu (Data Monitoring Committee) hay không

Nhóm Đặc điểm Can thiệp:
- `conditions`: Các tình trạng bệnh lý
- `num_conditions`: Số lượng điều kiện bệnh
- `intervention_types`: Loại can thiệp (DRUG, BIOLOGICAL, DEVICE, PROCEDURE, RADIATION, v.v.)
- `intervention_names`: Tên các can thiệp
- `num_interventions`: Số lượng can thiệp

Biến Mục tiêu:
- `label_binary`: Nhãn phân loại (1 = Thành công/COMPLETED, 0 = Thất bại/TERMINATED hoặc WITHDRAWN)

### 2.3. Kích Thước và Đặc Điểm Dữ Liệu

- Tổng số mẫu: 30,000 thử nghiệm lâm sàng
- Phân bổ nhãn: Khoảng 73% Thành công và 27% Thất bại (dữ liệu mất cân bằng)
- Phân chia dữ liệu: Train (70%) / Validation (10%) / Test (20%) với Stratified Sampling
- Tỷ lệ giá trị thiếu: Dao động từ 0% đến 5.4% tùy thuộc vào từng cột

---

## 3. Phương Pháp Nghiên Cứu

### 3.1. Quy Trình Xử Lý Dữ Liệu

Quy trình xử lý dữ liệu được thực hiện theo 4 giai đoạn chính:

Giai đoạn 1 - Thu thập dữ liệu (Notebook 01):
- Kiểm tra kết nối API và ước lượng số lượng thử nghiệm khả dụng
- Tải dữ liệu thô dạng JSON Lines với cơ chế retry để xử lý lỗi mạng
- Chuyển đổi và làm phẳng cấu trúc JSON lồng nhau thành định dạng CSV

Giai đoạn 2 - Tiền xử lý dữ liệu (Notebook 02):
- Chọn lọc 23 cột cố định theo schema định trước
- Phân tích và xử lý giá trị thiếu với các chiến lược phù hợp từng loại biến
- Tạo biến mục tiêu label_binary dựa trên trạng thái thử nghiệm
- Phân chia Train/Validation/Test với Stratified Sampling

Giai đoạn 3 - Phân tích khám phá (Notebook 03):
- Phân tích phân phối và thống kê mô tả cho từng biến
- Thực hiện kiểm định thống kê (Chi-square, Mann-Whitney U, Kruskal-Wallis)
- Trực quan hóa mối quan hệ giữa các đặc trưng và biến mục tiêu
- Trả lời 5 câu hỏi nghiên cứu về các yếu tố ảnh hưởng

Giai đoạn 4 - Feature Engineering (Notebook 04):
- Tạo các đặc trưng dẫn xuất (duration_days, enrollment_per_arm, is_late_phase, v.v.)
- Mã hóa biến phân loại (One-Hot Encoding, Target Encoding)
- Chuẩn hóa biến số (Standard Scaling)
- Loại bỏ các features có nguy cơ data leakage (has_results, enrollment_met)

### 3.2. Thuật Toán Sử Dụng

Dự án triển khai và so sánh 5 thuật toán phân loại:

Dummy Classifier:
- Mục đích: Thiết lập baseline performance
- Chiến lược: Dự đoán theo phân phối lớp đa số (most_frequent)

Logistic Regression:
- Mô hình tuyến tính với regularization L2
- Ưu điểm: Diễn giải dễ dàng thông qua hệ số hồi quy
- Hyperparameters: C=1.0, max_iter=1000

Random Forest:
- Ensemble method sử dụng kỹ thuật Bagging
- Xử lý mất cân bằng lớp qua class_weight='balanced'
- Hyperparameters: n_estimators=100, max_depth=10

XGBoost:
- Gradient Boosting với tối ưu hóa hiệu suất
- Sử dụng scale_pos_weight để xử lý imbalanced data
- Hyperparameters sau tuning: n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8

LightGBM:
- Gradient Boosting tối ưu cho dữ liệu lớn
- Sử dụng class_weight='balanced'
- Hyperparameters: n_estimators=200, max_depth=6, learning_rate=0.1

### 3.3. Chiến Lược Đánh Giá

Metrics đánh giá được lựa chọn phù hợp với bài toán dữ liệu mất cân bằng:

Metric chính:
- PR-AUC (Precision-Recall Area Under Curve): Đánh giá khả năng phân biệt trên lớp thiểu số
- Recall(Fail): Khả năng phát hiện các trường hợp thất bại

Metrics bổ sung:
- F1(Fail): Cân bằng giữa Precision và Recall cho lớp Fail
- ROC-AUC: Khả năng phân biệt tổng thể
- MCC (Matthews Correlation Coefficient): Metric cân bằng cho dữ liệu lệch
- Balanced Accuracy: Trung bình Recall của hai lớp

Tối ưu ngưỡng quyết định:
- Max F1 Strategy: Tìm ngưỡng tối ưu hóa F1-Score
- Youden's J Statistic: Cân bằng Sensitivity và Specificity
- Cost-Sensitive Strategy: Xem xét chi phí của False Positive và False Negative

---

## 4. Cài Đặt và Thiết Lập

### 4.1. Yêu Cầu Hệ Thống

- Python: Phiên bản 3.11 trở lên
- Hệ điều hành: Windows, macOS, hoặc Linux

### 4.2. Cài Đặt Môi Trường

Bước 1 - Clone repository:
```bash
git clone https://github.com/b28dbca4/Clinical-Trial-Failure-Prediction.git
cd Clinical-Trial-Failure-Prediction
```

Bước 2 - Tạo môi trường ảo với Conda:
```bash
conda env create -f environment.yml
conda activate clinical-trial-env
```

Hoặc sử dụng pip:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 4.3. Các Thư Viện Chính

- pandas (2.1.1): Xử lý và phân tích dữ liệu dạng bảng
- numpy (1.26): Tính toán số học và mảng đa chiều
- scikit-learn (1.3.1): Thuật toán học máy và đánh giá mô hình
- xgboost (2.0): Gradient Boosting hiệu suất cao
- lightgbm (4.1): Gradient Boosting tối ưu cho dữ liệu lớn
- matplotlib (3.8) và seaborn (0.13): Trực quan hóa dữ liệu
- shap (0.44): Giải thích mô hình học máy
- plotly (5.18): Trực quan hóa tương tác
- requests (2.31): Gọi API thu thập dữ liệu

---

## 5. Hướng Dẫn Sử Dụng

### 5.1. Chạy Toàn Bộ Pipeline

Để thực hiện toàn bộ quy trình phân tích, chạy các notebook theo thứ tự:

```bash
jupyter lab
```

Hoặc:

```bash
jupyter notebook
```

### 5.2. Chi Tiết Từng Notebook

Notebook 01 - Thu thập dữ liệu (01_data_collection.ipynb):
- Mục đích: Kết nối API ClinicalTrials.gov và tải dữ liệu thô
- Input: Tham số cấu hình API (từ khóa, phase, status)
- Output: data/raw/ctgov_raw.jsonl, data/processed/ctgov_flat.csv
- Thời gian chạy ước tính: 10-30 phút (phụ thuộc kết nối mạng)

Notebook 02 - Tiền xử lý dữ liệu (02_data_preprocessing.ipynb):
- Mục đích: Làm sạch dữ liệu và chuẩn bị cho phân tích
- Input: data/processed/ctgov_flat.csv
- Output: data/processed/clean_data.csv, train/val/test splits
- Thời gian chạy ước tính: 2-5 phút

Notebook 03 - Phân tích khám phá (03_eda_and_questions.ipynb):
- Mục đích: Trả lời 5 câu hỏi nghiên cứu thông qua EDA
- Input: data/processed/clean_data.csv
- Output: Các biểu đồ trong reports/figures/eda_plots/
- Thời gian chạy ước tính: 2-5 phút

Notebook 04 - Feature Engineering (04_feature_engineering.ipynb):
- Mục đích: Tạo đặc trưng và chuẩn bị dữ liệu cho modeling
- Input: data/processed/modeling/train_data.csv, val_data.csv, test_data.csv
- Output: train_features.csv, val_features.csv, test_features.csv
- Thời gian chạy ước tính: 2-5 phút

Notebook 05 - Huấn luyện mô hình (05_modeling.ipynb):
- Mục đích: Xây dựng, đánh giá và so sánh các mô hình học máy
- Input: data/processed/modeling/*_features.csv
- Output: models/best_model.joblib, model_config.json
- Thời gian chạy ước tính: 5-15 phút

Notebook 06 - Tổng kết (06_summary.ipynb):
- Mục đích: Tóm tắt kết quả và đưa ra khuyến nghị
- Input: Kết quả từ các notebook trước
- Output: Báo cáo tổng hợp

---

## 6. Kết Quả

### 6.1. Kết Quả Đánh Giá Mô Hình

Hiệu suất các mô hình trên tập Test (với ngưỡng mặc định 0.5 để so sánh công bằng):

```
Mô hình              ROC-AUC   Accuracy   Precision   Recall   F1-Score
--------------------|---------|----------|-----------|--------|----------
XGBoost (Tuned)     | 0.8635  | 0.8200   | 0.8833    | 0.8679 | 0.8755
XGBoost             | 0.8621  | 0.8175   | 0.8816    | 0.8661 | 0.8738
Random Forest       | 0.8614  | 0.8292   | 0.8693    | 0.9013 | 0.8850
LightGBM            | 0.8603  | 0.8195   | 0.8823    | 0.8684 | 0.8753
Logistic Regression | 0.8368  | 0.7955   | 0.8849    | 0.8272 | 0.8551
```

Mô hình XGBoost (Tuned) được chọn làm mô hình cuối cùng với ngưỡng quyết định tối ưu (Max F1 Strategy). Kết quả cuối cùng trên Test set:
- ROC-AUC: 0.8635
- PR-AUC với khoảng tin cậy 95%
- Threshold tối ưu được chọn theo chiến lược Max F1
- MCC (Matthews Correlation Coefficient) để đánh giá hiệu suất trên dữ liệu mất cân bằng

### 6.2. Feature Importance

Top 10 đặc trưng quan trọng nhất theo SHAP values:

1. `enrollment_per_arm`: Số lượng tuyển dụng mỗi nhánh nghiên cứu
2. `enrollment_count`: Tổng quy mô tuyển dụng
3. `sponsor_class_INDUSTRY`: Nhà tài trợ từ khối công nghiệp
4. `is_late_phase`: Thử nghiệm Phase 3
5. `has_dmc_True`: Có Ủy ban Giám sát Dữ liệu
6. `duration_days`: Thời gian thực hiện thử nghiệm
7. `allocation_RANDOMIZED`: Phân bổ ngẫu nhiên
8. `num_conditions`: Số lượng điều kiện bệnh lý
9. `masking_DOUBLE`: Thiết kế mù đôi
10. `intervention_type_DRUG`: Can thiệp bằng thuốc

### 6.3. Các Phát Hiện Chính từ EDA

Câu hỏi 1 - Tác động của Nhà tài trợ:
- Nhà tài trợ Industry có tỷ lệ thành công cao hơn đáng kể so với Academic/Research
- Sự hiện diện của DMC làm tăng tỷ lệ thành công thử nghiệm
- Chất lượng quản trị và giám sát là yếu tố then chốt

Câu hỏi 2 - Thời gian và Chiến lược Fail Fast:
- Phần lớn thử nghiệm thất bại bộc lộ vấn đề trong 12-24 tháng đầu
- Phase 2 đóng vai trò "cửa ải" quan trọng nhất
- Chiến lược phát hiện sớm thất bại giúp tiết kiệm nguồn lực đáng kể

Câu hỏi 3 - Chất lượng Thiết kế:
- Thiếu thông tin (Unknown) về thiết kế là chỉ báo mạnh cho nguy cơ thất bại
- Thiết kế Randomized và Factorial có tỷ lệ thành công cao hơn
- Sự minh bạch trong hồ sơ thiết kế quan trọng hơn loại hình thiết kế

Câu hỏi 4 - Độ Phức tạp:
- Thử nghiệm càng phức tạp (nhiều điều kiện, nhiều can thiệp) thì nguy cơ thất bại càng cao
- Thử nghiệm Device và Procedure có tỷ lệ thành công cao hơn Drug và Biological
- Đơn giản hóa thiết kế giúp tăng khả năng hoàn thành

Câu hỏi 5 - Xu hướng Thời gian:
- Không có suy thoái dài hạn trong ngành R&D ung thư
- Các đợt tăng tỷ lệ thành công trùng với làn sóng công nghệ mới (Immunotherapy giai đoạn 2011-2014)
- Đổi mới công nghệ là động lực chính phá vỡ rào cản nghiên cứu

### 6.4. Trực Quan Hóa Kết Quả

Các biểu đồ kết quả được lưu trong thư mục reports/figures/ bao gồm:
- eda_plots/: Biểu đồ phân tích khám phá (phân phối, tương quan, so sánh nhóm)
- model_results/: Đường cong ROC, PR Curve, Confusion Matrix, SHAP Summary Plot

---

## 7. Cấu Trúc Dự Án

```
Clinical-Trial-Failure-Prediction/
├── data/
│   ├── raw/                          # Dữ liệu thô từ API
│   │   ├── ctgov_raw.jsonl           # Dữ liệu JSON Lines gốc
│   │   ├── ctgov_metadata.json       # Metadata quá trình thu thập
│   │   └── logs/                     # Log files
│   │
│   ├── processed/                    # Dữ liệu đã xử lý
│   │   ├── ctgov_flat.csv            # Dữ liệu đã làm phẳng
│   │   ├── clean_data.csv            # Dữ liệu đã làm sạch
│   │   ├── data_dictionary.csv       # Từ điển dữ liệu
│   │   ├── data_dictionary.json      # Từ điển dữ liệu (JSON)
│   │   ├── split_manifest.json       # Thông tin phân chia dữ liệu
│   │   └── modeling/                 # Dữ liệu cho modeling
│   │       ├── train_data.csv        # Tập huấn luyện
│   │       ├── val_data.csv          # Tập validation
│   │       ├── test_data.csv         # Tập kiểm thử
│   │       ├── *_features.csv        # Dữ liệu sau feature engineering
│   │       ├── feature_dictionary.json # Từ điển/miêu tả các feature sau FE
│   │       └── fe_params.json        # Tham số feature engineering
│   │
│   └── final/                        # Dữ liệu cuối cùng
│           ├── models/
│           ├── best_model.joblib             # Mô hình tốt nhất đã lưu
│           ├── model_config.json             # Cấu hình mô hình (loại model, tham số, đường dẫn feature,...)
│           └── clinical_trial_model_*.joblib # Các phiên bản mô hình theo lần chạy / thời điểm
│
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Thu thập dữ liệu từ API
│   ├── 02_data_preprocessing.ipynb   # Tiền xử lý và làm sạch
│   ├── 03_eda_and_questions.ipynb    # Phân tích khám phá và câu hỏi nghiên cứu
│   ├── 04_feature_engineering.ipynb  # Tạo đặc trưng
│   ├── 05_modeling.ipynb             # Huấn luyện và đánh giá mô hình
│   └── 06_summary.ipynb              # Tổng kết và khuyến nghị
│
├── reports/
│   ├── figures/                      # Biểu đồ và hình ảnh xuất ra
│   │   ├── eda_plots/                # Biểu đồ EDA
│   │   └── model_results/            # Hình/kết quả đánh giá mô hình (ROC, PR, confusion matrix,...)
│   └── monitoring_report_*.json      # Báo cáo monitoring theo từng lần chạy / batch
│
├── src/
│   ├── __init__.py                   # Khởi tạo package src
│   ├── data/                         # Module xử lý dữ liệu
│   │   ├── __init__.py               # Khởi tạo subpackage data
│   │   ├── data_loader.py            # Tải/đọc dữ liệu (raw/processed)
│   │   ├── data_cleaner.py           # Làm sạch dữ liệu
│   │   └── feature_engineering.py    # Sinh feature, biến đổi, lưu feature + params
│   │
│   ├── models/                       # Module mô hình
│   │   ├── __init__.py               # Khởi tạo subpackage models
│   │   ├── model_trainer.py          # Huấn luyện mô hình
│   │   ├── model_evaluator.py        # Đánh giá mô hình
│   │   ├── threshold_optimizer.py    # Tối ưu ngưỡng quyết định 
│   │   ├── calibration_utils.py      # Hiệu chuẩn xác suất 
│   │   └── advanced_evaluation.py    # Đánh giá nâng cao 
│   │
│   ├── visualization/                # Module trực quan hoá
│   │   ├── __init__.py               # Khởi tạo subpackage visualization
│   │   ├── plot_utils.py             # Hàm vẽ chung 
│   │   └── dashboard.py              # Dashboard tương tác để theo dõi kết quả
│   │
│   └── utils/                        # Module tiện ích dùng chung
│       ├── __init__.py               # Khởi tạo subpackage utils
│       ├── config.py                 # Cấu hình chung (path, seed, param mặc định, env vars)
│       └── monitoring.py             # Giám sát & logging (run log, data drift, model drift,...)
│
├── environment.yml                   # Môi trường Conda (python + dependencies + channels)
├── requirements.txt                  # Danh sách dependencies cho pip (phiên bản gợi ý)
├── README.md                         # Tài liệu dự án (mục tiêu, setup, hướng dẫn chạy)
└── LICENSE                           # Giấy phép Apache 2.0

```

---

## 8. Thách Thức và Giải Pháp

### 8.1. Thách Thức về Dữ Liệu

Thách thức 1 - Dữ liệu mất cân bằng:
- Vấn đề: Tỷ lệ lớp 73% Thành công và 27% Thất bại gây khó khăn cho việc học lớp thiểu số.
- Giải pháp: Sử dụng `class_weight='balanced'`, scale_pos_weight trong XGBoost, và lựa chọn metric đánh giá phù hợp (PR-AUC, F1-Score thay vì Accuracy).

Thách thức 2 - Giá trị thiếu cao:
- Vấn đề: Nhiều biến quan trọng có tỷ lệ giá trị thiếu đáng kể (Unknown).
- Giải pháp: Phân tích pattern missing data, áp dụng các chiến lược điền giá trị phù hợp (median cho biến số, mode cho biến phân loại), và giữ lại category "UNKNOWN" như một chỉ báo quan trọng.

Thách thức 3 - Data Leakage:
- Vấn đề: Một số features như `has_results` và `enrollment_met` chỉ có giá trị sau khi thử nghiệm kết thúc.
- Giải pháp: Xác định và loại bỏ các features có nguy cơ leakage, chỉ sử dụng thông tin có sẵn tại thời điểm thử nghiệm bắt đầu (pre-trial features).

### 8.2. Thách Thức về Kỹ Thuật

Thách thức 4 - Cấu trúc JSON phức tạp:
- Vấn đề: Dữ liệu từ API có cấu trúc JSON lồng nhau nhiều cấp.
- Giải pháp: Xây dựng pipeline flatten tự động, lưu trữ dữ liệu thô ở định dạng JSON Lines để bảo toàn thông tin gốc.

Thách thức 5 - Tối ưu hyperparameter:
- Vấn đề: Không gian tìm kiếm lớn với nhiều tham số cần điều chỉnh.
- Giải pháp: Sử dụng GridSearchCV với PredefinedSplit để tận dụng tập validation sẵn có, tập trung vào các hyperparameter quan trọng nhất.

### 8.3. Thách Thức về Phân Tích

Thách thức 6 - Survivorship Bias:
- Vấn đề: Thử nghiệm thất bại thường kết thúc sớm hơn, gây nhiễu trong phân tích thời gian.
- Giải pháp: Nhận thức rõ hạn chế này khi diễn giải kết quả, tập trung vào các yếu tố có thể đo lường tại thời điểm bắt đầu thử nghiệm.

---

## 9. Hướng Phát Triển

### 9.1. Mở Rộng Nguồn Dữ Liệu

- Kết hợp dữ liệu từ các nguồn bổ sung như PubMed, FDA reports, và Patent databases để có cái nhìn đa chiều hơn về các thử nghiệm.
- Thu thập thêm thông tin về chi phí, mức độ tuân thủ điều trị, và các chỉ số sinh học cụ thể.

### 9.2. Cải Tiến Kỹ Thuật

- Ứng dụng NLP và Large Language Models (LLM) để trích xuất thông tin chi tiết từ các trường văn bản tự do (tiêu chí lựa chọn, mô tả chi tiết can thiệp).
- Triển khai các mô hình Deep Learning như Neural Networks và Transformer-based models.
- Xây dựng hệ thống dự đoán real-time với API và giao diện web.

### 9.3. Phân Tích Nâng Cao

- Áp dụng phân tích Survival Analysis để mô hình hóa thời gian đến sự kiện thất bại.
- Xây dựng hệ thống Risk Scoring tổng hợp từ tất cả các yếu tố để hỗ trợ ra quyết định.
- Thực hiện nghiên cứu định tính kết hợp phỏng vấn chuyên gia để lý giải sâu hơn các phát hiện định lượng.

### 9.4. Ứng Dụng Thực Tiễn

- Phát triển dashboard tương tác cho người dùng cuối (nhà nghiên cứu, công ty dược phẩm).
- Tích hợp với quy trình đánh giá thử nghiệm lâm sàng hiện có.
- Mở rộng phạm vi nghiên cứu sang các lĩnh vực bệnh khác ngoài ung thư.

---

## 10. Tác Giả

### 10.1. Thông Tin Nhóm

Dự án được thực hiện bởi nhóm sinh viên Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM, trong khuôn khổ môn học Nhập môn Khoa học Dữ liệu (Introduction to Data Science).

Thành viên 1:
- Họ và tên: Lăng Phú Quý
- MSSV: 23120415
- Vai trò: Phân tích EDA, Xây dựng mô hình, Feature Engineering

Thành viên 2:
- Họ và tên: Ngô Thị Thục Quyên
- MSSV: 23120348
- Vai trò: Thu thập dữ liệu, Tiền xử lý, Trực quan hóa

### 10.2. Liên Hệ

- Email: 23120415@student.hcmus.edu.vn
- Email: 23120348@student.hcmus.edu.vn
- Repository: https://github.com/b28dbca4/Clinical-Trial-Failure-Prediction
---

## 11. Giấy Phép

Dự án này được phát hành theo giấy phép Apache License 2.0.

Giấy phép cho phép:
- Sử dụng cho mục đích thương mại và phi thương mại
- Sửa đổi và phân phối lại mã nguồn
- Sử dụng riêng tư

Yêu cầu:
- Ghi rõ nguồn gốc và bản quyền
- Đính kèm giấy phép khi phân phối
- Ghi chú các thay đổi đã thực hiện

Dữ liệu từ ClinicalTrials.gov được sử dụng theo chính sách công khai của U.S. National Institutes of Health (NIH), phù hợp cho mục đích nghiên cứu và giáo dục.

---

Tài liệu tham khảo:
1. ClinicalTrials.gov API Documentation: https://clinicaltrials.gov/api/gui
2. NIH Data Sharing Policy: https://grants.nih.gov/policy/sharing.htm
3. Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
4. XGBoost Documentation: https://xgboost.readthedocs.io/