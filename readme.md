# build image:
    docker-compose build
    docker-compose up -d

check for errors with 
    docker-compose logs web

connect the the docker postgres
    docker-compose exec db psql -U personal_rag_user -d personal_rag_db
    exit with ctr + d


# run image:
docker run -p 8000:8000 personal_rag


# test
python3 manage.py runserver  



# setup .env file for docker :
    DB_NAME=personal_rag_db
    DB_USER=personal_rag_user
    DB_PASSWORD=Your_password
    DB_HOST=db
    DB_PORT=5432
    llama_cloud_api=llx-JHA************
    OPENAI_API_KEY=sk-proj-qh-*********
    ENVIRONMENT_TYPE=DEV 
    ENFORCE_SSL=true

