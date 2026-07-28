@echo off

chcp 65001

cd /d D:\AI_stock_predict


echo ============================== >> logs\startup.log
echo AI Stock Predict Start >> logs\startup.log
echo %date% %time% >> logs\startup.log
echo ============================== >> logs\startup.log


D:\conda_envs\stock_ai\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 >> logs\uvicorn.log 2>&1