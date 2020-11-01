import os

os.system(
    "docker run --restart=always -d --name redis_1 \
   -v /opt/redis/etc/redis.conf:/usr/local/etc/redis/redis.conf \
   -v /opt/redis/data:/data \
   -p 127.0.0.1:6379:6379 redis redis-server"
)
