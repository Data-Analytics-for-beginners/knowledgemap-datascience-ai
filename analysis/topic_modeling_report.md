# Тематичне моделювання: TF-IDF + LDA

Звіт згенеровано скриптом `analysis/topic_modeling.py` (перегенерувати: `python analysis/topic_modeling.py`).

## Параметри запуску

- Корпус: `data/processed/articles_clean.csv` — 1478 статей (текст = `title` + `excerpt`)
- Векторизація: tfidf, n-грами [1, 2], min_df=5, max_df=0.5, словник 1146 термів
- Модель: scikit-learn `LatentDirichletAllocation`, `learning_method=batch`, `random_state=42`
- Обрана кількість тем: **8**

## Підбір кількості тем

Метрика — UMass coherence (внутрішня когерентність теми: наскільки часто ключові слова теми зустрічаються разом в одних статтях). Чим ближче до 0, тим краще. Perplexity наведено довідково.

| k | Coherence (UMass) | Perplexity |
| --- | --- | --- |
| 8 ✅ | -1.947 | 3123.5 |

## Теми

### Тема 0. Обробка природної мови

- Статей: **119** (8.1% корпусу)
- Ключові слова: `model`, `language`, `language model`, `large`, `image`, `ai`, `large language`, `llm`, `scale`, `meta`
- Теги статей теми: Deep Learning (6 ст., lift 4.49), Generative AI (11 ст., lift 4.37), OpenAI (6 ст., lift 3.81), ChatGPT (5 ст., lift 2.89), Large Language Models (22 ст., lift 2.24)
- Приклади статей:
  - [What is Llama 3? The Experts' View on The Next Generation of Open Source LLMs](https://www.datacamp.com/blog/meta-announces-llama-3-the-next-generation-of-open-source-llms)
  - [Enhancing Large Language Models with Knowledge Graphs](https://www.datacamp.com/blog/knowledge-graphs-and-llms)
  - [Promoting Responsible AI: Content Moderation in ChatGPT](https://www.datacamp.com/blog/promoting-responsible-ai-content-moderation-in-chatgpt)

### Тема 1. Хмарні платформи та Data Engineering

- Статей: **234** (15.8% корпусу)
- Ключові слова: `difference`, `cloud`, `data`, `power`, `bi`, `power bi`, `aws`, `learn`, `course`, `case`
- Теги статей теми: Power BI (18 ст., lift 3.64), AWS (17 ст., lift 3.33), Cloud (39 ст., lift 2.91), Data Engineering (45 ст., lift 1.91), Python (13 ст., lift 1.09)
- Приклади статей:
  - [The Best Data Science Courses to Take in 2026](https://www.datacamp.com/blog/best-data-science-courses)
  - [8 of The Most Popular Machine Learning Tools](https://www.datacamp.com/blog/most-popular-machine-learning-tools)
  - [AWS vs Azure vs GCP: Which to Learn for Data Engineering](https://www.datacamp.com/blog/aws-vs-azure-vs-gcp)

### Тема 2. Кар'єра та навчання в Data Science

- Статей: **522** (35.3% корпусу)
- Ключові слова: `data`, `science`, `data science`, `skill`, `guide`, `learning`, `learn`, `ai`, `discover`, `project`
- Теги статей теми: Data Analysis (62 ст., lift 1.76), Career Services (89 ст., lift 1.65), Python (38 ст., lift 1.49), Data Science (96 ст., lift 1.48), For Business (42 ст., lift 1.42)
- Приклади статей:
  - [What is Data Literacy? A 2026 Guide for Data & Analytics Leaders](https://www.datacamp.com/blog/why-data-literacy-is-important-for-your-team)
  - [What is Data Literacy? A 2026 Guide for Data & Analytics Leaders](https://www.datacamp.com/blog/why-is-data-literacy-important)
  - [What is Data Literacy? A 2026 Guide for Data & Analytics Leaders](https://www.datacamp.com/blog/the-complete-guide-to-data-literacy)

### Тема 3. Підготовка до співбесід

- Статей: **94** (6.4% корпусу)
- Ключові слова: `interview`, `question`, `interview question`, `answer`, `question answer`, `top`, `advance`, `cover`, `basic`, `prepare`
- Теги статей теми: SQL (9 ст., lift 3.05), Career Services (28 ст., lift 2.64), Data Engineering (25 ст., lift 2.52), Cloud (11 ст., lift 1.95), Python (9 ст., lift 1.79)
- Приклади статей:
  - [Top 37 Azure Data Engineering Interview Questions for 2026](https://www.datacamp.com/blog/azure-data-engineering-interview-questions)
  - [Top 25 MongoDB Interview Questions and Answers for 2026](https://www.datacamp.com/blog/mongodb-interview-questions)
  - [Top 30 Data Structure Interview Questions and Answers for 2026](https://www.datacamp.com/blog/data-structure-interview-questions)

### Тема 4. Кар'єра та навчання в Data Science (spotlight)

- Статей: **119** (8.1% корпусу)
- Ключові слова: `data`, `ai`, `spotlight`, `literacy`, `upskill`, `employee`, `report`, `business`, `datalab`, `analyst`
- Теги статей теми: Life at DataCamp (24 ст., lift 4.24), AI for Business (7 ст., lift 3.11), For Business (21 ст., lift 2.9), Data Literacy (22 ст., lift 2.26), Learner Stories (9 ст., lift 2.25)
- Приклади статей:
  - [Bridging the Communications Gap with Data Literacy](https://www.datacamp.com/blog/bridging-the-communications-gap-with-data-literacy)
  - [What’s Driving the Data Literacy Skills Gap?](https://www.datacamp.com/blog/what-s-driving-the-data-literacy-skills-gap)
  - [25 Practical Examples of AI Transforming Industries](https://www.datacamp.com/blog/examples-of-ai)

### Тема 5. Освітні партнерства та благодійність

- Статей: **73** (4.9% корпусу)
- Ключові слова: `donate`, `classroom`, `student`, `partner`, `free`, `education`, `scholarship`, `year`, `teacher`, `data`
- Теги статей теми: DataCamp Classrooms (21 ст., lift 12.99), DataCamp Donates (33 ст., lift 9.52), Learner Stories (13 ст., lift 4.69), Product News (9 ст., lift 3.54), Data Literacy (22 ст., lift 3.26)
- Приклади статей:
  - [Reflecting on Another Year of Social Impact: DataCamp Classrooms 2023-2024 Annual Report](https://www.datacamp.com/blog/datacamp-classrooms-annual-report-2023-2024)
  - [DataCamp for Classrooms is Now Free to Belgian Secondary School Teachers and Students](https://www.datacamp.com/blog/datacamp-for-classrooms-is-now-free-to-belgian-secondary-school-teachers-and-students)
  - [DataCamp Classrooms Annual Report 2022](https://www.datacamp.com/blog/datacamp-classrooms-annual-report-2022)

### Тема 6. Порівняння AI-моделей

- Статей: **180** (12.2% корпусу)
- Ключові слова: `claude`, `benchmark`, `code`, `model`, `ai`, `feature`, `compare`, `agent`, `price`, `openai`
- Теги статей теми: AI News (13 ст., lift 6.15), AI Agents (29 ст., lift 4.96), OpenAI (11 ст., lift 4.42), Large Language Models (65 ст., lift 4.18), ChatGPT (8 ст., lift 2.92)
- Приклади статей:
  - [Claude Opus 4.7 vs Gemini 3.1 Pro: Which Model Is Better?](https://www.datacamp.com/blog/claude-opus-4-7-vs-gemini-3-1-pro)
  - [Claude Opus 4.8 vs Gemini 3.5 Flash: Benchmarks and Use Cases Compared](https://www.datacamp.com/blog/gemini-3-5-flash-vs-claude-opus-4-8)
  - [Claude Opus 5 vs Claude Fable 5: Which Anthropic Model Should You Use?](https://www.datacamp.com/blog/claude-opus-5-vs-claude-fable-5)

### Тема 7. Великі мовні моделі та генеративний AI

- Статей: **137** (9.3% корпусу)
- Ключові слова: `ai`, `machine`, `learning`, `machine learning`, `database`, `generative`, `example`, `generative ai`, `learn`, `application`
- Теги статей теми: Machine Learning (33 ст., lift 3.22), SQL (8 ст., lift 2.06), Generative AI (5 ст., lift 1.77), Artificial Intelligence (77 ст., lift 1.71), AI Agents (6 ст., lift 1.44)
- Приклади статей:
  - [AI Agents Business Applications: Examples, Benefits, Challenges](https://www.datacamp.com/blog/ai-agents-business-applications)
  - [Understanding AI Agents: The Future of Autonomous Systems](https://www.datacamp.com/blog/ai-agents)
  - [What can you do with SQL?](https://www.datacamp.com/blog/what-can-you-do-with-sql)

## Звірка з таксономією тегів

Теги (`tags` в `articles_clean.csv`) — це людська таксономія DataCamp, тож вони працюють як приблизна «земля правди» для тем, знайдених LDA. Для кожного тега в темі рахується **lift** — у скільки разів тег частіший усередині теми, ніж у корпусі загалом:

- `lift ≈ 1` — тег однаково поширений усюди (напр. наскрізний `Artificial Intelligence`) і нічого не підтверджує;
- `lift > 2` — тема справді «зловила» окрему частину таксономії.

Нижче — теми, відсортовані за найсильнішим lift їхнього провідного тега.

| Тема | Провідний тег | Lift | Частка статей теми з тегом |
| --- | --- | --- | --- |
| 5. Освітні партнерства та благодійність | DataCamp Classrooms | 12.99 | 14% |
| 6. Порівняння AI-моделей | AI News | 6.15 | 4% |
| 0. Обробка природної мови | Deep Learning | 4.49 | 3% |
| 4. Кар'єра та навчання в Data Science (spotlight) | Life at DataCamp | 4.24 | 11% |
| 1. Хмарні платформи та Data Engineering | Power BI | 3.64 | 4% |
| 7. Великі мовні моделі та генеративний AI | Machine Learning | 3.22 | 14% |
| 3. Підготовка до співбесід | SQL | 3.05 | 5% |
| 2. Кар'єра та навчання в Data Science | Data Analysis | 1.76 | 7% |

## Наступний крок

`data/processed/topics.json` уже містить усе, що потрібно для графа (`viz/`): `topics[].topic_id` — вузли, `edges` — зважені ребра за косинусною подібністю розподілів слів. Альтернативний спосіб побудувати ребра — колонки `topic_id` / `topic_second_id` у `data/processed/articles_with_topics.csv`: скільки статей поєднують дві теми.
