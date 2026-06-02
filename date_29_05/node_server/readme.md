docker build -t local-web-app:v1 .

docker run -d -p 8080:8080 --name test-app local-web-app:v1
