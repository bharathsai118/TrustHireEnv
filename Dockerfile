FROM python:3.11-slim

LABEL maintainer="TrustHireEnv Contributors"
LABEL description="OpenEnv-compliant multimodal interview integrity benchmark"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "\
    from env.environment import TrustHireEnv; \
    env = TrustHireEnv(difficulty='easy'); \
    obs = env.reset(); \
    _, _, done, info = env.step({'flag_level': 'none', 'next_step': 'continue'}); \
    print('Docker smoke-test OK'); \
    "

CMD ["python", "baseline_eval.py", "--no-llm", "--episodes", "3"]