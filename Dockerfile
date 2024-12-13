# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.13

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /QML-Toolbox
COPY requirements.txt /QML-Toolbox/requirements.txt

# Install pip requirements
RUN python -m pip install --no-cache-dir -r /QML-Toolbox/requirements.txt

COPY . /QML-Toolbox

ENTRYPOINT [ "python" ]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /QML-Toolbox
USER appuser

# Install Rust using rustup
ENV RUST_VERSION=stable
ENV PATH="/home/appuser/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION

RUN rustc --version && cargo --version

WORKDIR /QML-Toolbox/zx-calculus
RUN cargo build --release
WORKDIR /QML-Toolbox


# Expose the port that Uvicorn will run on
EXPOSE 8000

CMD ["main.py", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]