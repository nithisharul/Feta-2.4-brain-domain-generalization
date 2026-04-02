#!/usr/bin/env python
#notes on install
#python -m pip install --upgrade pip
#python -m pip install itk-morphologicalcontourinterpolation

import itk
import pip
import sys

input_filename = sys.argv[1]
output_filename = sys.argv[2]
#radius = sys.argv[3]

image = itk.imread(input_filename)

processed = itk.MorphologicalContourInterpolator.New(Input=image)
#itk.MorphologicalContourInterpolator(image,output_filename)

itk.imwrite(processed, output_filename)
