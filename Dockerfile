FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /dpo_test

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/dpo_test
CMD ["python", "src/main.py"]