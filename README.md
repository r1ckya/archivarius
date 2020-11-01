# Архивариус

## Run project:
``` sudo docker/run.sh --init ```


## Rest API
Post file:

``` curl -i -X POST -F "file=@tmp" localhost:5000/api/upload ```


Edit file:

``` curl -d '{"process_status":"failed"}' -H "Content-Type: application/json" -X POST http://localhost:5000/api/edit/2280273608 ```

Get file:


``` curl localhost:5000/api/upload/6965061037 ```

Search:

```
curl -d '{"doc_id":"3345218050"}' -H "Content-Type: application/json" -X POST http://localhost:5000/api/search
```

Run MySQL:
``` sudo docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=1234 -d mysql:latest ```
