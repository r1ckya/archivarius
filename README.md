# poedim

Run project:
``` sudo docker/run.sh --init ```

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

Run redis:
``` docker run --restart=always -d --name redis_1 \
   -v /opt/redis/etc/redis.conf:/usr/local/etc/redis/redis.conf \
   -v /opt/redis/data:/data \
   -p 127.0.0.1:6379:6379 redis redis-server ```

Run MySQL:
``` sudo docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=1234 -d mysql:latest ```
