# Финальное задание по Deep Learning Basic

## Постановка задачи
Необходимо реализовать модель OCR для распознавания капчи.
Структура модели была взята из следующей [статьи](https://arxiv.org/pdf/1507.05717.pdf), посвященной 
распознаванию последовательностей на изображениях. Основная идея заключается в извлечении 
признаком с помощью глубокой сверточной сети, а далее их трансформации в последовательность символов.
Так как в датасете все капчи состояли из одинакогого количества
символов, можно было использовать кросс-энтропийную функцию потерь.


## Запуск обучения и оценки
Из корневой директории репозитория `./` выполнить:
```
python source/train.py \
--model_path checkpoints/model.pth \
--data_path samples/
```
В выводе содержится итоговое качество на тестовой выборке, а так же объекты, на которых моделей 
ошибается сильнее всего.

## Итоговая метрика
В качестве метрики использовалась доля неверно распознанных 
символов, `Character Error Rate (CER)`. Финальная метрика на тесте: `CER=0.05`.

## Анализ ошибок
Модель преимущественно ошибается на примерах с повторяющимися символами, а также иногда 
не различает некоторые символы, которые похожи и идут подряд, например m и n. Чтобы решить эту проблему, 
можно попробовать обучить другую модель только на этих примерах, и далее использовать ансамбль.