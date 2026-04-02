#last change 4.2.2020
#
#Used for single subject or batch processing
#
#Usage:
#$1 is directory name
#$2 is the grayscale image
#
#Requirements: C3D, FSL, python, ITK
#
#Andras Jakab, 2020, University of Zurich
#andras.jakab@kispi.uzh.ch

function logo { 
	echo " "
	echo "----------------------------------------------------------------------------------------------"
	echo "|     Processing of manually annotated labels, FetA2020 dataset   Step 1. Process labels      |"
	echo "|                                                                                             |"
	echo "|                                           Andras Jakab,  University of Zurich               |"
	echo "----------------------------------------------------------------------------------------------"
	echo " "
}

function Usage { 
    logo
    echo "Requirements: Requirements: C3D, FSL, python, ITK" 
    echo " " 
    echo "-----------------------------------------------------------------------------------------------------------"                             
    echo "Usage:"
    echo " "  
    echo "Processlabels.sh (1) (2)"
    echo " "  
    echo "(1): input directory where annotated labels are found, full path "  
    echo "(2): grayscale image annotated, full path "  
    echo " "  
    echo " "  
    echo "Usage notes: "  
    echo " "  
    echo "More documentation: andras.jakab@kispi.uzh.ch"  
    echo "-----------------------------------------------------------------------------------------------------------" 
    exit 1
}

[ "$1" = "" ] && Usage
[ "$1" = "-help" ] && Usage
[ "$1" = "-h" ] && Usage
[ "$1" = "--help" ] && Usage

#c3d install directory, change when migrating system
export PATH=$PATH:/usr/local/c3d/
export PATH=$PATH:/usr/local/c3d/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/c3d/

dir=$1
cd $1 
#Renames annotated label maps according to actual label value, see help
#Important that labels are saved accordingly, and the naming contains -label as shown below
origimage=$dir/*_processed.nii.gz

for file in $(ls *processed*-label*.nii.gz); do 
	echo $file; p=$(fslstats $file -M)
	fname=$(echo $p | awk '{print int($1+0.5)}')
	cp $file label_"$fname".nii.gz
	#silences output in case no such files are found by ls
done &> /dev/null

for file in $(ls *processed*_label*.nii.gz); do 
	echo $file; p=$(fslstats $file -M)
	fname=$(echo $p | awk '{print int($1+0.5)}')
	cp $file label_"$fname".nii.gz 
done &> /dev/null

for file in $(ls 7.nii.gz); do 
	echo $file; p=$(fslstats $file -M)
	fname=$(echo $p | awk '{print int($1+0.5)}')
	cp $file label_"$fname".nii.gz
done &> /dev/null

#Creates background image (every voxel)
fslmaths $2 -thr 0.01 -bin -mul 4 label_background

#Does the file exist?
function checkfile {
	if [ -f $1 ]; 
		then
		#schlafen!
		sleep 0
		else
		echo "Previous pre-processing step did not complete successfully. File $1 for case in $dir is missing. Exiting."
		#exit 1
	fi
}

#Check if initial annotations exist after renaming
checkfile label_1.nii.gz
checkfile label_2.nii.gz
checkfile label_3.nii.gz
checkfile label_5.nii.gz
checkfile label_6.nii.gz
checkfile label_7.nii.gz
checkfile label_8.nii.gz

#functions for processing the annotated labels, specific for each tissue type

function label1 {
	#intracranial space
	cp label_1.nii.gz orig_label_1.nii.gz
	fslmaths label_1.nii.gz -mul 1 label_1.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation.py label_1.nii.gz label_1_proc.nii.gz 
	fslmaths label_1_proc.nii.gz -bin -mul 1 label_1_proc.nii.gz
}

function label2 {
	#cortex, w smoothing
	cp label_2.nii.gz orig_label_2.nii.gz
	fslmaths label_2.nii.gz -mul 1 label_2.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation_axial.py label_2.nii.gz label_2_proc.nii.gz 
	fslmaths label_2_proc.nii.gz -bin -s 0.5 -thr 0.5 -ero -bin -mul 2 label_2_proc.nii.gz
}

function label3 {
	#white matter, minimal smoothing, axial interpolation
	cp label_3.nii.gz orig_label_3.nii.gz
	fslmaths label_3.nii.gz -mul 1 label_3.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation_axial.py label_3.nii.gz label_3_proc.nii.gz 
	fslmaths label_3_proc.nii.gz -bin -s 0.75 -thr 0.5 -bin -mul 3 label_3_proc.nii.gz
}

function label5 {
	#ventricle system
	cp label_5.nii.gz orig_label_5.nii.gz
	fslmaths label_5.nii.gz -mul 1 label_5.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation_axial.py label_5.nii.gz label_5_proc.nii.gz 
	fslmaths label_5_proc.nii.gz -bin -mul 15 label_5_proc.nii.gz
}

function label6 {
	#cerebellum
	cp label_6.nii.gz orig_label_6.nii.gz
	fslmaths label_6.nii.gz -mul 1 label_6.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation.py label_6.nii.gz label_6_proc.nii.gz
	fslmaths label_6_proc.nii.gz -s 0.25 -thr 0.6 -bin -mul 6 label_6_proc.nii.gz
}

function label7 {
	#basal ganglia and thalamus, some smoothing
	cp label_7.nii.gz orig_label_7.nii.gz
	fslmaths label_7.nii.gz -mul 1 label_7.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation_axial.py label_7.nii.gz label_7_proc.nii.gz 
	fslmaths label_7_proc.nii.gz -bin -s 1 -thr 0.4 -bin -dilM -mul 7 label_7_proc.nii.gz
}

function label8 {
	#brainstem
	cp label_8.nii.gz orig_label_8.nii.gz
	fslmaths label_8.nii.gz -mul 1 label_8.nii.gz -odt char
	python /media/m-ssd1/feta/morphologicalcontourinterpolation.py label_8.nii.gz label_8_proc.nii.gz
	fslmaths label_8_proc.nii.gz -bin -mul 8 label_8_proc.nii.gz
}

#processes annotated labels in parallel
label1 &
label2 &
label3 &
label5 &
label6 &
label7 &
label8 &
wait

#check if output files exist
checkfile label_1_proc.nii.gz
checkfile label_2_proc.nii.gz
checkfile label_3_proc.nii.gz
checkfile label_5_proc.nii.gz
checkfile label_6_proc.nii.gz
checkfile label_7_proc.nii.gz
checkfile label_8_proc.nii.gz

#combines labels 
c3d label_1_proc.nii.gz label_2_proc.nii.gz label_3_proc.nii.gz label_6_proc.nii.gz label_7_proc.nii.gz label_8_proc.nii.gz label_5_proc.nii.gz -accum -max -endaccum -o combined.nii.gz
fslmaths combined.nii.gz -thr 15 -uthr 15 15.nii.gz
fslmaths combined.nii.gz -sub 15.nii.gz temp2.nii.gz
fslmaths 15.nii.gz -div 3 -add temp2.nii.gz combined.nii.gz
fslmaths combined.nii.gz -bin -mul 4 temp.nii.gz
fslmaths label_background -sub temp.nii.gz temp.nii.gz
fslmaths combined.nii.gz -add temp.nii.gz combined.nii.gz
fslmaths combined.nii.gz -thr 0.001 -uthr 8.001 combined.nii.gz

#PADS files to 256x256x256  (useful for DL applications)
xdim=$(fslinfo combined.nii.gz | grep dim1 | awk 'NR==1{print $2}')
ydim=$(fslinfo combined.nii.gz | grep dim2 | awk 'NR==1{print $2}')
zdim=$(fslinfo combined.nii.gz | grep dim3 | awk 'NR==1{print $2}')
xpad=$(( 256 - $xdim ))
ypad=$(( 256 - $ydim ))
zpad=$(( 256 - $zdim ))
#padding of combined and T2-SR volumes 
c3d combined.nii.gz -pad 0x0x0vox "$xpad""x""$ypad""x""$zpad"vox 0 -o combined.nii.gz
c3d $origimage -pad 0x0x0vox "$xpad""x""$ypad""x""$zpad"vox 0 -o T2_SR_padded.nii.gz

#cleans up processing directory
rm temp.nii.gz
rm temp2.nii.gz
rm *_proc.nii.gz
rm *background.nii.gz
rm 15.nii.gz
rm label_*.nii.gz

#creates separate label files
label=1; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=2; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=3; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=4; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=5; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=6; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=7; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
label=8; fslmaths combined.nii.gz -thr $label -uthr $label label_$label.nii.gz &
wait
exit 1


