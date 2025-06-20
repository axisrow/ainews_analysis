# AI News Research Project

Научный проект для исследования новостей об искусственном интеллекте в интернете.

## Описание

Этот проект автоматически собирает, анализирует и визуализирует новости об ИИ из различных источников:
- Новостные сайты (web scraping)
- RSS фиды технологических изданий

## Возможности

- **Сбор данных** из множества источников
- **NLP анализ**: извлечение ключевых фраз, сущностей, тем
- **Анализ тональности** новостей с использованием современных ML моделей
- **GenAI анализ**: глубокий анализ с помощью Google Gemini (опционально)
- **Управление ключевыми словами**: интеллектуальное извлечение и редактирование
- **Интерактивный дашборд**: визуализация трендов и редактирование данных
- **База данных** для хранения и поиска по собранным материалам

## Установка

### Вариант 1: Docker (рекомендуется)

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd ainews_analysis

# Соберите Docker образ
docker build -t ainews-analysis .

# Запустите контейнер
docker run -v $(pwd)/data:/home/app/data ainews-analysis python main.py --help
```

### Вариант 2: Локальная установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd ainews_analysis
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Скачайте языковую модель spaCy:
```bash
python -m spacy download en_core_web_sm
```

4. (Опционально) Создайте .env файл для API ключей:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

## Использование

### Локально

```bash
# Полный цикл: сбор -> анализ -> отчет
python main.py

# Только сбор новых данных
python main.py --collect

# Анализ существующих данных
python main.py --analyze

# Анализ ключевых слов
python main.py --keywords

# Генерация отчета
python main.py --report

# Запуск дашборда
python main.py --dashboard
```

### Docker

```bash
# Сбор данных
docker run -v $(pwd)/data:/home/app/data -v $(pwd)/.env:/home/app/.env ainews-analysis python main.py --collect

# Анализ данных
docker run -v $(pwd)/data:/home/app/data -v $(pwd)/.env:/home/app/.env ainews-analysis python main.py --analyze

# Анализ ключевых слов
docker run -v $(pwd)/data:/home/app/data ainews-analysis python main.py --keywords

# Запуск дашборда
docker run -p 8501:8501 -v $(pwd)/data:/home/app/data ainews-analysis python main.py --dashboard
```

### Дополнительные опции

```bash
# Тестовый режим (ограниченное количество статей)
python main.py --test --collect

# Повторный анализ всех статей
python main.py --analyze --reanalyze

# Множественный запуск
python main.py --collect --multi-run 5
```

## Структура проекта

```
ainews_analysis/
├── src/
│   ├── scrapers/         # Модули сбора данных
│   │   ├── web_scraper.py
│   │   ├── rss_scraper.py
│   │   └── date_extractor.py
│   ├── analyzers/        # NLP и AI анализ
│   │   ├── nlp_analyzer.py
│   │   ├── sentiment_analyzer.py
│   │   ├── genai_analyzer.py
│   │   └── keyword_analyzer.py
│   ├── models/           # База данных
│   │   └── database.py
│   ├── utils/            # Утилиты
│   │   └── url_normalizer.py
│   └── visualization/    # Дашборд
│       └── dashboard.py
├── data/                 # Хранение данных
│   ├── ai_news.db       # SQLite база данных
│   └── keywords.json    # Ключевые слова
├── logs/                 # Логи
├── reports/              # Отчеты
├── config.py            # Конфигурация
├── main.py              # Главный скрипт
├── dashboard.py         # Точка входа для дашборда
├── requirements.txt     # Зависимости
├── Dockerfile           # Docker образ
├── .dockerignore        # Docker исключения
└── .env                 # Переменные окружения (создать вручную)
```

## Конфигурация

Настройки проекта находятся в `config.py`:
- Источники данных (сайты, RSS)  
- Ключевые слова для поиска
- Параметры NLP и GenAI анализа
- Настройки базы данных

### Переменные окружения (.env)

```bash
# Google GenAI API ключ (опционально)
GOOGLE_API_KEY=your_api_key_here
```

## Источники данных

### Web Scraping
- AI News (artificialintelligence-news.com)
- Analytics Insight AI
- Другие AI новостные сайты

### RSS Feeds  
- TechCrunch AI
- Wired AI
- MIT Technology Review
- VentureBeat AI
- The Verge AI

## Анализ

### NLP Анализ
- **Sentiment Analysis**: определение тональности с помощью transformer моделей и TextBlob
- **Entity Recognition**: извлечение организаций, персон, продуктов через spaCy
- **Topic Modeling**: автоматическое определение тем
- **Keyword Extraction**: интеллектуальное извлечение ключевых слов с TF-IDF и NER

### GenAI Анализ (Google Gemini)
- **Глубокий анализ содержания**: значимость, сложность, инновационность
- **Извлечение инсайтов**: ключевые идеи и области влияния
- **Классификация**: исследовательский vs коммерческий фокус
- **Fallback режим**: работает без API ключа

### Дашборд
- **Интерактивная визуализация**: графики, тренды, статистика
- **Редактор ключевых слов**: добавление, удаление, поиск и фильтрация
- **Анализ по источникам**: сравнение различных новостных каналов
- **Временные тренды**: отслеживание изменений во времени

## Требования

- Python 3.11+ (рекомендуется)
- 4GB RAM минимум (8GB для больших объемов данных)
- Интернет-соединение для сбора данных
- Docker (опционально, но рекомендуется)

## Лицензия

MIT