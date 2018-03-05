@echo off
setlocal enabledelayedexpansion

set main_dir=C:\Users\husby036\Documents\Cprojects\test_s2s\russia_central_east\matlab\tif_results\2m
set link_dir=C:\Users\husby036\Documents\Cprojects\test_s2s\russia_central_east\python\tif_results\2m

for %%f in (!main_dir!\*) do (
  set "main_file=%%f"

  for %%i in (!main_file!) do (
    rem for /f "delims=_" %%a in ("%%~ni") do set prefix=%%a
    rem if "!prefix!" EQU "test" (
      set "link_file=!link_dir!\%%~ni%%~xi"
      if not exist !link_file! (
        mklink /h !link_file! !main_file!
      )
    rem )
  )
)
