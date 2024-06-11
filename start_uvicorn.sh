#!/bin/bash
source /app/IDM-VTON/venv/bin/activate
exec uvicorn app_VTON:app --host 0.0.0.0 --port 7860
