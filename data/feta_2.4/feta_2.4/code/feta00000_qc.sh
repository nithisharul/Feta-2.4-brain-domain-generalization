# Visual quality check script
# Version 4.2.2020
#
# Requirements: FSL, ImageMagick library for Linux
# 
# $1 is the subject directory where the results are found $2 is the subject ID, check batch script for explanation
# Places qc.png in the subject directory (montage: axial and sagittal views) and a subject-named qc file n the QC directory

#installation specific, change when migrating script
lut=$FSLDIR/etc/luts/customlut.lut 
outputdir=/media/fsl/m-ssd1/feta/all/qc
dir=$1
echo "Directory is: $dir"

#brainimage=$(find $dir/ -name "*_processed.nii.gz")
brainimage=$dir/T2_SR_padded.nii.gz
subid=$2

echo "Brain image is: $brainimage"
fslswapdim $dir/combined.nii.gz y z x $dir/combined_rot.nii.gz
fslswapdim $brainimage y z x $dir/brainimage_rot.nii.gz

slicer $dir/combined.nii.gz -l $lut -i 1 8 -S 15 2500 $dir/a.png
slicer $brainimage -i 50 1500 -S 15 2500 $dir/b.png

slicer $dir/combined_rot.nii.gz -l $lut -i 1 8 -S 15 2500 $dir/c.png
slicer $dir/brainimage_rot -i 50 1500 -S 15 2500 $dir/d.png

composite -dissolve 40 -gravity Center $dir/a.png $dir/b.png -alpha Set $dir/"$subid"_ax1.jpg
composite -dissolve 2 -gravity Center $dir/a.png $dir/b.png -alpha Set $dir/"$subid"_ax0.jpg
composite -dissolve 40 -gravity Center $dir/c.png $dir/d.png -alpha Set $dir/"$subid"_sag1.jpg
composite -dissolve 2 -gravity Center $dir/c.png $dir/d.png -alpha Set $dir/"$subid"_sag0.jpg
montage $dir/*_*.jpg -tile 1x4 -geometry +0+0 -gamma 1.2 -mattecolor black -bordercolor black $outputdir/$subid.png
cp $outputdir/$subid.png $dir/qc.png

rm $dir/"$subid"_*.jpg
rm $dir/a.png $dir/b.png $dir/c.png $dir/d.png $dir/brainimage_rot.nii.gz $dir/combined_rot.nii.gz
