
# Movie Reviews Sentiment Analysis

## Description

Ứng dụng phân tích cảm xúc bình luận phim là tích cực, tiêu cực hay trung tính dựa trên các mô hình deep learning RNN và LSTM, sử dụng framework Pytorch để tạo và huấn luyện mô hình, và được triển khai trên web framework Django.


## Get Started

Dữ liệu dùng để huấn luyện mô hình là các bình luận của người dùng được thu thập từ top 100 phim nổi bật trên [IMDB](https://www.imdb.com). Cách thu nhập dữ liệu từ các bộ phim trên IMDB liệt kê trong file [urls.txt](data/urls.txt) được mô tả trong [data_collecting.py](data/data_collecting.py) . Dữ liệu sau khi thu thập được tổng hợp thành 2 file dataset [train.tsv](data/reviews.tsv) và [test.tsv](data/test.tsv) với chi tiết các nhãn:
 * 0 (tiêu cực): Các bình luận có đánh giá từ 1 đến 4.
 * 1 (trung tính): Các bình luận có đánh giá từ 5 đến 7.
 * 2 (tích cực): Các bình luận có đánh giá từ 8 đến 10.

 Dữ liệu trong file [train.tsv](data/reviews.tsv) dùng để huấn luyện mô hình RNN và dữ liệu trong file [test.tsv](data/test.tsv) dùng để đánh giá độ chính xác của mô hình.

Dự án sử dụng thư viện Pytorch để tạo và huấn luyện mô hình. Để cài đặt Pytorch, xem hướng dẫn chi tiết trên [PyTorch website](https://pytorch.org/get-started/locally).


## Requirements

### Cài đặt Pytorch

Xem chi tiết các cài đặt trên [PyTorch website](https://pytorch.org/get-started/locally).

### Phiên bản Python
 * 3.9 <= Python <= 3.11


## Installation

Clone repo:

```
git clone https://github.com/tuannguyen8531/Movie-Reviews-Sentiment-Analysis.git
```
    
Khởi tạo và kích hoạt môi trường ảo (nếu cần):

```
cd Movie-Reviews-Sentiment-Analysis
python -m venv venv
```
* Windows
```
venv/Scripts/activate
```
* Linux
```
source venv/bin/Activate.ps1 
```

Cài đặt các dependency:

```
pip install -r requirements.txt
```

Cài đặt spacy english data:

```
python -m spacy download en_core_web_sm
```

Chạy project:

```
python manage.py runserver
```


## Notebook

Notebook [sentiment-analysis.ipynb](sentiment-analysis.ipynb) mô tả chi tiết các bước để tạo ra mô hình phân tích cảm xúc bình luận phim là tích cực, tiêu cực hay trung tính:

* Tiền xử lý dữ liệu
* Vector hóa dữ liệu
* Tạo và huấn luyện mô hình LSTM
* Phân tích cảm xúc bình luận
* Đánh giá mô hình

![Workflow](workflow.png)


## Screenshots

![Screenshot](screenshot.png)


## References

* <https://pytorch.org/docs/stable/index.html>
* <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>
* <https://www.analyticsvidhya.com/blog/2021/07/understanding-rnn-step-by-step-with-pytorch/>
* <https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/>
* <https://www.kaggle.com/competitions/sentiment-analysis-on-imdb-movie-reviews>
* <https://www.analyticsvidhya.com/blog/2022/01/sentiment-analysis-with-lstm/>
* <https://www.embedded-robotics.com/sentiment-analysis-using-lstm/>
* <https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L15/2_packed-lstm.ipynb>
