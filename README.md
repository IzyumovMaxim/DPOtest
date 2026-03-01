# Реализация Direct Preference Optimization

В данном проекте представлено решение тестового задания по ручной реализации алгоритма DPO на основе статьи "Direct Preference Optimization: Your Language Model is Secretly a Reward Model".

## Структура проекта

```text
.
├── Dockerfile          
├── requirements.txt  
├── theory.pdf          # решение теоретической части с кратким изложением статьи, анализом преимуществ DPO и разбором off-policy постановки задачи
└── src/
    ├── main.py         # точка входа для запуска всего пайплайна
    ├── data.py         # логика загрузки датасета и токенизации
    ├── model.py        # инициализация Policy-модели и замороженной Reference-модели
    ├── dpo_loss.py     # кастомная математическая реализация функции потерь DPO
    └── training.py     # цикл обучения
```

## Инструкция по воспроизведению результатов

### Запуск через Docker
1. Собираем Docker-образ:
   ```bash
   docker build -t dpo_project .
   ```
2. Запускаем контейнер на сервере с GPU:
   ```bash
    docker run --gpus all -it dpo_project
    ```
### Локальная установка
1. Устанавливаем зависимости:
    ```
   pip install -r requirements.txt
   ```
2. Запускаем обучение:
     ```
    python -m src.main
   ```

## Анализ результатов
