FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /dpo_test

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]