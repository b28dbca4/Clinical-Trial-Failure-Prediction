# Phân tích Rủi ro và Dự đoán Thất bại trong Thử nghiệm Lâm sàng

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)

Dự án này áp dụng các kỹ thuật Khoa học dữ liệu và Học máy để phân tích các yếu tố rủi ro và xây dựng mô hình dự đoán khả năng thất bại của các thử nghiệm lâm sàng, nhằm hỗ trợ ra quyết định chiến lược trong nghiên cứu dược phẩm.

---

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Phương pháp nghiên cứu (Methodology)](#phương-pháp-nghiên-cứu-methodology)
4. [Cài đặt & Thiết lập](#cài-đặt--thiết-lập)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Kết quả (Results)](#kết-quả-results)
7. [Cấu trúc dự án](#cấu-trúc-dự-án)
8. [Thách thức & Giải pháp](#thách-thức--giải-pháp)
9. [Hướng phát triển](#hướng-phát-triển)
10. [Tác giả & Liên hệ](#tác-giả--liên-hệ)

---

## Giới thiệu

### Mô tả bài toán
Thử nghiệm lâm sàng là giai đoạn quan trọng nhưng tốn kém nhất trong quy trình phát triển thuốc. Tỷ lệ thất bại cao dẫn đến lãng phí nguồn lực khổng lồ. Bài toán đặt ra là làm thế nào để tận dụng dữ liệu lịch sử để xác định sớm các thử nghiệm có nguy cơ thất bại cao.

### Động lực và ứng dụng thực tế
Việc dự đoán chính xác kết cục thử nghiệm mang lại giá trị to lớn:
- **Tối ưu hóa chi phí:** Loại bỏ sớm các hướng nghiên cứu không khả thi.
- **An toàn bệnh nhân:** Giảm thiểu rủi ro cho người tham gia thử nghiệm.
- **Chiến lược đầu tư:** Hỗ trợ các công ty dược phẩm và nhà đầu tư đưa ra quyết định dựa trên dữ liệu.

### Mục tiêu cụ thể
- Thu thập và làm sạch dữ liệu từ các kho lưu trữ thử nghiệm công khai.
- Khám phá các yếu tố chính ảnh hưởng đến thành công (Feature Importance).
- Xây dựng mô hình phân loại (Classification Model) để dự đoán trạng thái: "Thất bại" hoặc "Thành công".
- Đưa ra các khuyến nghị kinh doanh (Business Insights) dựa trên kết quả mô hình.

---

## Dataset

### Nguồn dữ liệu
Dữ liệu được thu thập tự động (crawling/API) từ các cơ sở dữ liệu thử nghiệm lâm sàng uy tín (ví dụ: ClinicalTrials.gov).

### Mô tả các features
Tập dữ liệu bao gồm các đặc trưng đa dạng:
- **Thông tin hành chính:** Mã định danh, nhà tài trợ, ngày bắt đầu/kết thúc.
- **Thiết kế nghiên cứu:** Giai đoạn (Phase), loại nghiên cứu (Study Type), mô hình can thiệp.
- **Tiêu chí tham gia:** Điều kiện bệnh lý, tiêu chí chọn/loại (Inclusion/Exclusion criteria).
- **Quy mô:** Số lượng người tham gia (Enrollment).

### Đặc điểm dữ liệu
- Dữ liệu thô chứa nhiều nhiễu, giá trị thiếu và định dạng văn bản phi cấu trúc.
- Có sự mất cân bằng giữa các lớp (số lượng ca thành công thường ít hơn hoặc chênh lệch so với thất bại/bị hủy).

---

## Phương pháp nghiên cứu (Methodology)

Dự án tuân theo quy trình chuẩn CRISP-DM:

1.  **Thu thập dữ liệu (Data Collection):** Sử dụng script tự động để tải dữ liệu thô.
2.  **Tiền xử lý (Preprocessing):**
    *   Làm sạch dữ liệu: Xử lý giá trị Null, chuẩn hóa định dạng ngày tháng.
    *   Xử lý văn bản: Làm sạch các trường mô tả (text cleaning) để chuẩn bị cho trích xuất đặc trưng.
3.  **Phân tích khám phá (EDA):** Sử dụng các biểu đồ thống kê để hiểu phân phối dữ liệu và mối tương quan giữa các biến số với kết quả thử nghiệm.
4.  **Kỹ thuật đặc trưng (Feature Engineering):**
    *   *Encoding:* Chuyển đổi biến phân loại (Categorical) sang dạng số (One-hot, Label encoding).
    *   *NLP Features:* Trích xuất thông tin từ văn bản mô tả thử nghiệm (ví dụ: TF-IDF hoặc Embeddings đơn giản).
    *   *Scaling:* Chuẩn hóa dữ liệu số để tối ưu hóa hội tụ của thuật toán.
5.  **Huấn luyện mô hình (Model Training):**
    *   Thử nghiệm các thuật toán học máy giám sát (Supervised Learning) như Logistic Regression, Random Forest, Gradient Boosting (XGBoost/LightGBM).
    *   Sử dụng Cross-validation để đảm bảo tính tổng quát hóa.
6.  **Đánh giá (Evaluation):** So sánh hiệu năng dựa trên các metrics phù hợp với bài toán phân loại (đặc biệt chú trọng Precision/Recall do tính chất bài toán y tế).

---

## Cài đặt & Thiết lập

Yêu cầu: Python 3.11+ và Conda.

1.  **Clone repository:**
    ```bash
    git clone https://github.com/b28dbca4/Clinical-Trial-Failure-Prediction.git
    cd Clinical-Trial-Failure-Prediction
    ```

2.  **Thiết lập môi trường ảo:**
    ```bash
    conda env create -f environment.yml
    conda activate ds-env
    # Hoặc sử dụng pip
    pip install -r requirements.txt
    ```

---

## Hướng dẫn sử dụng

Khởi chạy Jupyter Notebook:
```bash
jupyter notebook
```

---

## Kết quả (Results)


---

## Cấu trúc dự án

```
├── data/
│   ├── raw/          # Dữ liệu thô ban đầu
│   ├── processed/    # Dữ liệu đang xử lý
│   └── final/        # Dữ liệu sạch dùng cho modeling
├── notebooks/        # Các bước phân tích tuần tự
├── src/              # Mã nguồn (modular code)
│   ├── data/         # Scripts xử lý dữ liệu
│   ├── models/       # Định nghĩa và huấn luyện model
│   └── visualization/# Tiện ích vẽ biểu đồ
├── reports/          # Báo cáo và hình ảnh kết quả
└── environment.yml   # Cấu hình môi trường
```

---

## Thách thức & Giải pháp

1.  **Dữ liệu thiếu và không đồng nhất:**
    *   *Giải pháp:* Sử dụng các kỹ thuật điền khuyết (Imputation) thông minh dựa trên phân nhóm, và loại bỏ các đặc trưng có tỷ lệ thiếu quá cao không mang lại giá trị thông tin.
2.  **Mất cân bằng dữ liệu (Imbalanced Data):**
    *   *Giải pháp:* Áp dụng kỹ thuật lấy mẫu lại (Resampling) như SMOTE hoặc điều chỉnh trọng số lớp (Class Weights) trong hàm mất mát của mô hình.
3.  **Xử lý dữ liệu văn bản y khoa:**
    *   *Giải pháp:* Sử dụng các thư viện NLP chuyên dụng để loại bỏ từ dừng (stopwords) và chuẩn hóa từ vựng chuyên ngành.

---

## Hướng phát triển

*   **Deep Learning:** Áp dụng các mô hình ngôn ngữ lớn (Transformers/BERT) để hiểu sâu hơn ngữ nghĩa trong phần mô tả thử nghiệm và tiêu chí.
*   **Time-series Analysis:** Phân tích xu hướng thay đổi của các thử nghiệm theo thời gian.
*   **Deployment:** Đóng gói mô hình thành API hoặc Web App để người dùng cuối có thể nhập thông số thử nghiệm và nhận dự báo rủi ro theo thời gian thực.

---

## Tác giả & Liên hệ

Mọi đóng góp, thắc mắc hoặc báo lỗi vui lòng tạo Issue trên GitHub repository này.

---

## License

Dự án được cấp phép theo tiêu chuẩn [Apache 2.0 License](LICENSE).
