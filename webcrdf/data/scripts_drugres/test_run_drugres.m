close all;
clear all;

fnCT='example_0/inputct.nii.gz';
fnXR='example_0/inputxr.png';
fnCTSegm='example_0/inputct.nii.gz_maskct.nii.gz';
fnXRSegm='example_0/inputxr.png_maskxr.png';

wdir='.';

DrugResistorScript(wdir, fnXR, fnXRSegm, fnCT, fnCTSegm);

