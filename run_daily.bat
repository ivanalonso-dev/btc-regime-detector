@echo off
REM Actualiza datos, detector, registro en vivo y dashboard. Programalo una vez al dia.
cd /d "%~dp0"
python fetch_data.py || exit /b 1
python regime_detector.py || exit /b 1
python dashboard.py
