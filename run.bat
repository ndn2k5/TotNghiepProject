@echo off
title HR Policy Chatbot - Web
echo.
echo  ========================================
echo   Vietnamese HR Policy Chatbot
echo   Starting web server... please wait
echo  ========================================
echo.

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONPATH=.

echo  Opening browser at http://localhost:8501
echo  Press Ctrl+C to stop the server
echo.

start "" "http://localhost:8501"
streamlit run streamlit_app.py

pause
