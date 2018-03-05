rem ***** THIS IS A WORK IN PROGRESS *****

@echo off
setlocal enabledelayedexpansion

if [%1]==[] (
    set main_dir="C:\Users\husby036\Documents\Git\setsm_postprocessing_python"
) else (
    set main_dir=%1
)

if [%2]==[] (
    set link_dir="."
) else (
    set link_dir=%2
)

for %%f in (!main_dir!\*) do (
    set "main_file=%%f"
    for %%i in ("!main_file!") do (
        rem for /f "delims=_" %%a in ("%%~ni") do (set prefix=%%a)
        rem if "!prefix!"=="test" (
            set link_file=!link_dir!"\%%~ni%%~xi"
            if not exist !link_file! (
                mklink /h !link_file! "!main_file!"
            )
        rem )
    )
)

for /d %%d in ("!main_dir!\*") do (
    set "main_subdir=%%~fd"
    for %%i in ("!main_subdir!") do (
        if not "%%~nxd"==".idea" (
            set link_subdir=!link_dir!"\%%~ni%%~xi"
            if not exist !link_subdir! (
                mkdir !link_subdir!
                echo DIRECTORY CREATED: !link_subdir!
            )
            call "%~f0" "!main_subdir!" !link_subdir!
        )
    )
)
