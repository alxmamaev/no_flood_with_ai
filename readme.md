# Решение задачи по предсказанию уровня реки АМУР
По всем ошибкам запуска (но если набор данных будет такой, какой я ожидаю, то все будет ок) можно писать в телеграм @alxmamaev

## Запуск

```
docker build -t no_flood_with_ai .
docker run --volume $(pwd)/datasets:/usr/src/app/datasets no_flood_with_ai 2013-10-11 2013-10-21
```
