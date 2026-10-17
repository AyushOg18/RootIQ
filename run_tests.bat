@echo off
title RootIQ - Test Suite
echo ========================================================
echo Running RootIQ Automated Test Suite (pytest)
echo ========================================================
echo.
python -m pytest tests/ -v
pause
