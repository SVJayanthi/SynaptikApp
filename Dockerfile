FROM python:3.9

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install netcat dependencies
RUN apt-get update && apt-get install -y netcat

COPY requirements.txt .

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY entrypoint.sh ./entrypoint.sh

# run entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# gunicorn
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "run:app"]
