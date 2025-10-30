@echo off
echo Cleaning LaTeX auxiliary files...
echo.

del /S /Q *.log *.aux *.dvi *.lof *.lot *.bit *.idx *.glo *.bbl *.bcf *.ilg *.toc *.ind *.out *.blg *.fdb_latexmk *.fls *.run.xml *.synctex* *.acn *.ist 2>nul

echo.
echo Cleanup complete!
pause