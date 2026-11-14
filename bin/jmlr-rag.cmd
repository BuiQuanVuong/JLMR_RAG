@echo off
rem Windows launcher: runs jmlr-rag from this project's .venv without activating it.
rem Add this bin folder to your PATH (or copy this file into a folder that is on PATH and
rem change PROJECT below to the project folder), then type `jmlr-rag` in any terminal.
set "PROJECT=%~dp0.."
"%PROJECT%\.venv\Scripts\jmlr-rag.exe" %*
