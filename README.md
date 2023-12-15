# các bước chạy question 3:

git clone https://github.com/phanphong13/assignment.git
pip install -r requirements.txt
#Trong file .env sửa open_api_key

#crawl: chạy xong sẽ lưu 1 file data.txt ở trong thư mục web
#python crawl.py

#Indexing
#sẽ lưu trữ database vào thư mục vectorstores/db_faiss
#python compute_vector_db.py

Chạy truy vấn từ index đã lưu tại 3.1:

python run.py
