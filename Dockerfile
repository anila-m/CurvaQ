# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.13

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt


WORKDIR /QML-Toolbox
COPY . /QML-Toolbox


ENTRYPOINT [ "python" ]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /QML-Toolbox
USER appuser

# Expose the port that Uvicorn will run on
EXPOSE 8000

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["main.py", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]