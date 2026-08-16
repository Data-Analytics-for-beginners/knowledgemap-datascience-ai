# analysis/

NLP-аналіз тем на основі зібраних статей.

## Призначення

Виявити приховані теми та зв'язки між ними на основі заголовків, тегів та описів статей.

## Структура

| Файл | Статус | Призначення |
| --- | --- | --- |
| `clean_data.py` | ✅ готово | Очищення `data/raw/articles_raw.csv` → `data/processed/articles_clean.csv` |
| `data_stats.py` | ✅ готово | Precheck-звіт по очищеному датасету перед тематичним моделюванням |
| `text_preprocessing.py` | 🔜 план | Токенізація, стоп-слова, лематизація |
| `topic_modeling.py` | 🔜 план | TF-IDF + LDA (gensim) для виявлення тем |
| `tag_network.py` | 🔜 план | Матриця спільної появи тегів (co-occurrence) для графа |

## Очищення даних (`clean_data.py`)

```bash
python analysis/clean_data.py           # data/raw/articles_raw.csv → data/processed/
python analysis/clean_data.py -v        # + детальні логи
```

### Що робить скрипт

CSV читається через `pandas.read_csv` — повноцінний RFC 4180 парсер. Наївний
`line.split(",")` тут ламається: заголовки на кшталт
`"Gemini 3.7 Flash: Features, Benchmarks, and Pricing"` містять коми всередині лапок.

Фільтри застосовуються послідовно, кожен зі своєю причиною видалення:

1. **`missing url`** — рядок без URL (страховка).
2. **`missing published_date (certification page)`** — сторінки сертифікацій
   (`sql-certification`, `power-bi-certification`, `GitHub-certifications`).
   Вони живуть під `/blog/` і тому проходять URL-фільтр скрапера, але не мають
   ані `datePublished`, ані автора, ані тегів — саме порожня дата надійно
   відрізняє їх від статей.
3. **`unparsable published_date`** — дата не парситься як `%Y-%m-%d`. Парсинг
   навмисно строгий: `fetch_articles.py` вже нормалізує дати, тож інший формат
   означає, що скрапер змінився, і це має бути помітно, а не «тихо виправлено».
4. **`empty title`** / **`empty excerpt`** — базова валідація текстових полів,
   з яких далі будується корпус для TF-IDF/LDA.
5. **`duplicate url`** — дедуплікація за `url` (лишається перший рядок). Це
   другий рубіж захисту: `fetch_articles.merge_rows` вже дедуплікує при зборі.

### Результати

| Файл | Вміст |
| --- | --- |
| `data/processed/articles_clean.csv` | Очищені статті + похідні колонки |
| `data/processed/articles_dropped.csv` | Видалені рядки з колонкою `drop_reason` |
| `data/processed/cleaning_report.json` | Лічильники: скільки рядків видалено і чому |

Схема `articles_clean.csv` — це вихідні колонки
(`title, url, published_date, author, tags, excerpt, scraped_at`) плюс похідні:

- `published_year` — рік публікації (`int`)
- `published_month` — місяць у форматі `YYYY-MM`
- `tag_count` — кількість тегів у рядку

Скрипт **ідемпотентний**: повторний запуск читає той самий `raw/` і дає
байт-у-байт ідентичний результат. Сирі дані ніколи не змінюються.

### Робота з тегами

У CSV колонка `tags` лишається у вигляді рядка з роздільником `|`. Це свідоме
рішення: записаний у комірку `"['a', 'b']"` довелося б розбирати через
`ast.literal_eval` — крихко й із втратами. Для аналізу є окремі функції:

```python
from analysis.clean_data import load_clean, split_tags, tag_frequencies

df = load_clean()                 # + tags_list (list[str]), published_at (datetime)
df["tags_list"].head()            # [['Deep Learning', 'Large Language Models'], ...]

split_tags(df["tags"])            # те саме для довільного DataFrame
tag_frequencies(df).head(20)      # топ-20 тегів за кількістю статей
```

`split_tags` повертає `[]` для статті без тегів (а не `[""]`), тож `.explode()`
і `len()` працюють коректно.

## Статистика після очищення (`data_stats.py`)

```bash
python analysis/data_stats.py
python analysis/data_stats.py --top 30 --output data/processed/data_stats.txt
```

Друкує текстовий precheck-звіт із чотирьох блоків:

1. **Cleaning summary** — скільки рядків видалено і з якої причини (читає
   `cleaning_report.json`), із прикладами URL.
2. **Corpus overview** — розмір корпусу, діапазон дат, довжина `title`/`excerpt`
   у словах (чи вистачить тексту для LDA).
3. **Publication dates** — розподіл публікацій за роками з ASCII-гістограмою та
   часткою статей за останні 12 місяців.
4. **Tags** — топ-20 тегів, покриття тегами, скільки тегів зустрічається лише в
   одній статті (довжина «хвоста» таксономії).

Запускати перед тематичним моделюванням: перекошений за періодом корпус або
надто довгий хвіст тегів змінюють інтерпретацію тем.

## Методологія (орієнтовний план)

1. Очищення даних (`clean_data.py`) + precheck (`data_stats.py`)
2. Попередня обробка тексту (заголовки + excerpt)
3. Векторизація (TF-IDF)
4. Тематичне моделювання (LDA), підбір оптимальної кількості тем
5. Інтерпретація тем — присвоєння змістовних назв кластерам
6. Побудова матриці зв'язків: які теми/теги часто зустрічаються разом

## Результат

- `../data/processed/articles_clean.csv` — очищений датасет для NLP
- `../data/processed/topics.json` — виявлені теми з ключовими словами
- `../data/processed/tag_cooccurrence.csv` — матриця зв'язків для побудови графа

## Тести

```bash
python tests/test_clean_data.py
```

Офлайн-перевірки `clean_data.py` на власних CSV-фікстурах: коми в лапках,
порожні поля як `""` (а не `NaN`), кожен фільтр, розбиття тегів, ідемпотентність.
