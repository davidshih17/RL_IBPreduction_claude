#!/bin/bash
# Compile the symmetry routing note (two passes for references).
cd /het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/notes
pdflatex -interaction=nonstopmode symmetry_inference_routing.tex 2>&1
pdflatex -interaction=nonstopmode symmetry_inference_routing.tex 2>&1
