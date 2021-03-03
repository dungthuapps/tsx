#!/bin/bash

# sudo bash ./svg2pdf.sh folder_path
# get path from command line
MYWD=$1

echo "List all files under $MYWD"
for file in $MYWD/*.svg
do
  name=${file##*/}
  base=${name%.*}.pdf
  # base=${file%.*}.pdf
  echo "$file -> $MYWD/pdf/$base"
  inkscape $file \
    --export-area-drawing \
    --export-type="pdf" \
    --export-filename=$MYWD/pdf/$base
done