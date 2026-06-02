@echo off
title NanoServe - go live
REM ============================================================
REM  ONE-CLICK SHARE for the NanoServe playground.
REM   - clears any stuck server/tunnel
REM   - starts the inference server on localhost:8000
REM   - opens your Cloudflare tunnel
REM  Public link (always the same):  https://slm.masicoltd.com
REM  Keep this window OPEN and the PC ON while sharing.
REM  Press Ctrl+C (or close this window) to stop the tunnel.
REM ============================================================

echo Clearing any old server / tunnel...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo Starting the inference server (loads the model)...
start "nanoserve-server" /min C:\SLM_v2\.venv\Scripts\python.exe -m uvicorn server:app --app-dir C:\SLM_v2\src --port 8000

echo Waiting 30s for the model to finish loading...
timeout /t 30 /nobreak >nul

echo.
echo ============================================================
echo   LIVE at:   https://slm.masicoltd.com
echo   Send that link to the recruiter.
echo   Keep this window open. Ctrl+C to stop sharing.
echo ============================================================
echo.
C:\SLM_v2\tools\cloudflared.exe tunnel --config C:\SLM_v2\tools\slm-tunnel.yml run
