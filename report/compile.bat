@echo off
cd /d "%~dp0"
xelatex -interaction=nonstopmode thesis.tex && xelatex -interaction=nonstopmode thesis.tex
echo Done. Open thesis.pdf
