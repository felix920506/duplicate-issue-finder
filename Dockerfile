FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY duplicate_issue_finder.py web_ui.py system_prompt.txt verifier_prompt.txt ./

EXPOSE 7860

CMD ["python", "web_ui.py", "--host", "0.0.0.0", "--port", "7860"]
