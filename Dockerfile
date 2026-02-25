FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/

# Default handler — overridden per-function in CDK
CMD ["claudestreet.handlers.sentinel.handler"]
