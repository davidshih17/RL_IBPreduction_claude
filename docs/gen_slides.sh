#!/bin/bash
cd /het/p4/dshih/jet_images-deep_learning/RL_IBPreduction_claude/docs
/cmsb/HETconda/miniconda3/bin/pandoc slides.md -t beamer -o slides.pdf \
    --pdf-engine=xelatex \
    -V monofont="DejaVu Sans Mono" \
    -H header.tex
echo "PDF generated: slides.pdf"
