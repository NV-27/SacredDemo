# Sacred Demonstration SberLoga 2021

Демонстрация возможностей фреймворка [Sacred](https://github.com/IDSIA/sacred) для организации ML-pipelines. Демонстрация проведена для сообщества SberLoga 2021.

# Навигация по репозиторию

:white_check_mark: [Слайды](https://github.com/NV-27/SacredDemo/blob/master/references/sacred.pdf) презентации;

:white_check_mark: [Пайплайны](https://github.com/NV-27/SacredDemo/tree/master/pipeline), используемые в демонстрации;
  - [базовый пайплайн](https://github.com/NV-27/SacredDemo/blob/master/pipeline/base_pipeline.py), без использования возможностей Sacred;
  - [базовый sacred-пайплайн](https://github.com/NV-27/SacredDemo/blob/master/pipeline/base_sacred_pipeline.py), с использованием минимальных возможностей Sacred;
  - [sacred-пайплайн](https://github.com/NV-27/SacredDemo/blob/master/pipeline/sacred_pipeline.py), с использованием основных возможностей Sacred: автозапуск эксперимента и автоматическое сохранение мета-информации;
  - [sacred-пайплайн с конфигурированием](https://github.com/NV-27/SacredDemo/blob/master/pipeline/sacred_pipeline_with_config.py), с использованием функции-конфигуратора эксперимента;
  - [sacred-пайплайн с внешним конфигом](https://github.com/NV-27/SacredDemo/blob/master/pipeline/sacred_pipeline_with_external_config.py).

:white_check_mark: [Исходные файлы](https://github.com/NV-27/SacredDemo/tree/master/src), используемые в ходе обучения моделей;

:white_check_mark: [Конфигурационные файлы](https://github.com/NV-27/SacredDemo/tree/master/configs), используемые в advanced-пайплайнах;
