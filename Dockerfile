FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app


# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение в образ
COPY . .

# Создаём кеш-папки и даём права на запись
RUN mkdir -p /app/cache/hub /app/cache/datasets /app/mpl_cache \
    && chmod -R 777 /app/cache /app/mpl_cache

# Устанавливаем переменные окружения для Hugging Face и Matplotlib
ENV HF_HOME=/app/cache
ENV HF_DATASETS_CACHE=/app/cache/hub
ENV TRANSFORMERS_CACHE=/app/cache/hub
ENV MPLCONFIGDIR=/app/mpl_cache

# Открываем порт
EXPOSE 8501

# Проверка состояния
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск S
CMD streamlit run streamlit_app.py --server.port=8501
