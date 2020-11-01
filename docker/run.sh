#!/bin/bash
docker build -t poedim docker
docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=1234 -d mysql:latest
docker run --rm -ti -v /var/www/uploads:/var/www/uploads -v `pwd`:`pwd` -w `pwd` --net="host" poedim python server.py $1
