chcp 65001
@echo off
REM conda activate python38
call C:\Users\max_focus\anaconda3\Scripts\activate.bat C:\Users\max_focus\anaconda3\envs\python38
REM run python file
call conda activate python38
streamlit run C:\Users\max_focus\600南方基金\cta合集\cta综合指数\补充双因子分析.py [ARGUMENTS]
Pause
