set err.log=U:\Documents\Programming\PycharmProjects\Prod\SnowLevel_Bot\err.log
rem --- where Conda probably lives ---
set "CONDA_HOME=%USERPROFILE%\miniconda3"
if not exist "%CONDA_HOME%\condabin\conda.bat" set "CONDA_HOME=%USERPROFILE%\AppData\Local\miniconda3"

@echo off
For /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
For /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
@echo Current Date And Time Is: %mydate%_%mytime% >> U:\Documents\Programming\PycharmProjects\Prod\SnowLevel_Bot\err.log 2>&1
@echo Starting...>> U:\Documents\Programming\PycharmProjects\Prod\SnowLevel_Bot\err.log 2>&1
rem --- activate and run ---
call "%CONDA_HOME%\condabin\conda.bat" activate plotly_python
python U:\Documents\Programming\PycharmProjects\Prod\SnowLevel_Bot\snow_level_plotter.py >> U:\Documents\Programming\PycharmProjects\Prod\SnowLevel_Bot\err.log 2>&1
@echo FINISHED
@echo *****************************************************************