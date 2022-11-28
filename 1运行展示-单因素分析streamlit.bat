chcp 65001
@echo off
REM conda activate python38
call C:\Users\max_focus\anaconda3\Scripts\activate.bat C:\Users\max_focus\anaconda3\envs\python38
REM run python file
call conda activate python38
streamlit run test_streamlit.py [ARGUMENTS]
Pause
