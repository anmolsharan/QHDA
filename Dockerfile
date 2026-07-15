FROM python:3.11

WORKDIR /app

# Copy all project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---------- DEBUG ----------
RUN echo "===== APP DIRECTORY ====="
RUN ls -lah /app

RUN echo "===== MODEL FILE ====="
RUN ls -lh /app/grad_best_model.h5

RUN echo "===== FILE TYPE ====="
RUN file /app/grad_best_model.h5 || true

RUN echo "===== FIRST 10 LINES ====="
RUN head -10 /app/grad_best_model.h5 || true
# ---------------------------

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]