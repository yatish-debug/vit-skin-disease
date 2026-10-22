@echo off
title DermViT - Skin Disease Detection Web App
echo ========================================================
echo   Starting DermViT Vision Transformer Web Application...
echo   URL: http://127.0.0.1:8000
echo ========================================================
set PYTHONIOENCODING=utf-8
call .venv\Scripts\activate.bat
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
pause
