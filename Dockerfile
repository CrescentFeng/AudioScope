FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    NO_PROXY=127.0.0.1,localhost,::1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
RUN python -m pip install 'setuptools>=68' wheel && \
    python -m pip install 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu && \
    python -c "import tomllib,subprocess,sys; p=tomllib.load(open('pyproject.toml','rb'))['project']; deps=p['dependencies']+sum([p['optional-dependencies'][x] for x in ['database','api','engineering','demo','transfer']],[]); subprocess.check_call([sys.executable,'-m','pip','install',*deps])"
COPY src ./src
RUN python -m pip install --no-deps --no-build-isolation .

COPY alembic.ini ./
COPY migrations ./migrations
COPY l4_app.py ./
COPY configs ./configs

EXPOSE 8000
CMD ["uvicorn", "audioscope.api:app", "--host", "0.0.0.0", "--port", "8000"]
