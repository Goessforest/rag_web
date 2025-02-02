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
