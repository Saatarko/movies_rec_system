# Используем официальный образ Python с поддержкой Streamlit
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /streamlit_app.py

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .


# Запускаем Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
