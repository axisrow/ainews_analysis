# AI News Research Project

Научный проект для исследования новостей об искусственном интеллекте в интернете.

## Описание

Этот проект автоматически собирает, анализирует и визуализирует новости об ИИ из различных источников:
- Новостные сайты (web scraping)
- RSS фиды технологических изданий
- Reddit сообщества об ИИ

## Возможности

- **Сбор данных** из множества источников без необходимости в API ключах
- **NLP анализ**: извлечение ключевых фраз, сущностей, тем
- **Анализ тональности** новостей с использованием современных ML моделей
- **Визуализация** трендов и статистики через интерактивный дашборд
- **База данных** для хранения и поиска по собранным материалам

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd ai_project
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Скачайте языковую модель spaCy:
```bash
python -m spacy download en_core_web_sm
```

## Использование

### Сбор и анализ данных

```bash
# Полный цикл: сбор -> анализ -> отчет
python main.py

# Только сбор новых данных
python main.py --collect

# Анализ существующих данных
python main.py --analyze

# Генерация отчета
python main.py --report
```

### Запуск дашборда

```bash
# Через main.py (автоматически запустит streamlit)
python main.py --dashboard

# Или напрямую через streamlit
streamlit run dashboard.py
```

### Массовый сбор данных

```bash
# Быстрый параллельный сбор (рекомендуется)
python bulk_collector.py --runs 10

# Загрузка исторических данных
python historical_data.py

# Традиционный способ
python main.py --bulk --collect --multi-run 5
```

## Структура проекта

```
ai_project/
├── src/
│   ├── scrapers/         # Модули сбора данных
│   ├── analyzers/        # NLP и sentiment анализ
│   ├── models/           # База данных
│   └── visualization/    # Дашборд
├── data/                 # Хранение данных
├── notebooks/            # Jupyter notebooks
├── config.yaml          # Конфигурация
├── main.py              # Главный скрипт
└── requirements.txt     # Зависимости
```

## Конфигурация

Настройки проекта находятся в `config.yaml`:
- Источники данных (сайты, RSS, Reddit)
- Ключевые слова для поиска
- Параметры анализа
- Настройки визуализации

## Источники данных

### Web Scraping
- MIT Technology Review
- VentureBeat AI
- The Verge AI
- Ars Technica AI

### RSS Feeds
- TechCrunch AI
- Wired AI
- AI News

### Reddit
- r/artificial
- r/MachineLearning
- r/deeplearning
- r/singularity

## Анализ

- **Sentiment Analysis**: определение тональности новостей (позитивная/негативная/нейтральная)
- **Entity Recognition**: извлечение организаций, персон, продуктов
- **Topic Modeling**: автоматическое определение тем
- **Trend Analysis**: отслеживание изменений во времени

## Требования

- Python 3.8+
- 4GB RAM минимум
- Интернет-соединение для сбора данных

## Лицензия

MIT