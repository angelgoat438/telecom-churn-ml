# Imagen base ligera con Python
FROM python:3.11-slim

# Evita prompts interactivos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos requirements e instalamos dependencias
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY app ./app
COPY models ./models
COPY data ./data
COPY assets ./assets

# Puerto por defecto de Streamlit
EXPOSE 8501

# Comando de arranque
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
