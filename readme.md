# build image:
docker buildx build --tag personal_rag .

# run image:
docker run -p 8000:8000 personal_rag


# test
python3 manage.py runserver  
