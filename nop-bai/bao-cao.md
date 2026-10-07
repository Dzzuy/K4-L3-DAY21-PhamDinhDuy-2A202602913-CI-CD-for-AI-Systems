# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phạm Đình Duy |
| MSSV | 2A202602913 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Dzzuy/K4-L3-DAY21-PhamDinhDuy-2A202602913-CI-CD-for-AI-Systems |
| Ngày nộp | 08/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 có F1 lớp dương cao nhất dù accuracy của lần 1 lớn hơn. Cây nhiều và sâu hơn cải thiện nhận diện lớp thu nhập cao; cấu hình 50 cây, learning rate thấp học chưa đủ. Quét ngưỡng cho F1 0.7368 tại 0.30, so với 0.7149 tại 0.50.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Khoảng 24,8% mẫu có thu nhập trên 50K. Dự đoán mọi mẫu là thu nhập thấp vẫn đạt accuracy 75,2% nhưng F1 lớp dương bằng 0. F1 kết hợp precision và recall cho lớp cần nhận diện; Quality Gate dùng F1 lớp dương với ngưỡng 0.65. Accuracy và F1 macro/weighted có thể che khuất việc bỏ sót lớp thiểu số.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi tương thích | Setuptools mới bỏ `pkg_resources`; SQLAlchemy 2.1 không hợp MLflow 2.13 | Ghim `setuptools<81`, `sqlalchemy<2.1`. |
| S3 ban đầu bị từ chối | IAM thiếu quyền bucket | Cấp policy hẹp cho bucket Day 21 và CI OIDC. |
| Release VM lỗi | SciPy mới không hợp NumPy 1.26 | Ghim SciPy 1.13.1 và triển khai qua SSM. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.736842 | 0.8600 |
| Bước 3 (thêm `train_batch2`) | 0.753731 | 0.8680 |

**Nhận xét:** Tăng dữ liệu huấn luyện từ 22.361 lên 44.722 mẫu giúp F1 lớp dương tăng 0,016889 và accuracy tăng 0,0080 trên cùng tập holdout. Batch mới cải thiện khả năng nhận diện thu nhập cao. [Run tự động từ commit dữ liệu `065716f`](https://github.com/Dzzuy/K4-L3-DAY21-PhamDinhDuy-2A202602913-CI-CD-for-AI-Systems/actions/runs/37662702341) hoàn thành cả bốn jobs.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: CI đọc lại run `d29c91c6bda64ded9c8b0c398af99a0f` và đối chiếu F1/accuracy.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét 0,10–0,90; ngưỡng 0,30 tăng F1 lên 0,7368.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Ghi ma trận nhầm lẫn và precision/recall từng lớp vào `outputs/detail.txt`.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: Quality Gate chặn release khi F1 mới thấp hơn model hiện hành trên S3.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Cảnh báo khi tỷ lệ lớp dương lệch hơn 5 điểm phần trăm so với 24,8%.
