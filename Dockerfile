FROM python:3.12.14-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .
COPY tests ./tests
CMD ["mosca", "solve", "sum_list", "--depth", "7"]
