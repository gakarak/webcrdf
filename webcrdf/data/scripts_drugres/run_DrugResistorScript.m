close all;
clear all;

% % xrorig = 'example_0/inputxr.dcm';
% % xrsegm = 'example_0/inputxr.dcm_maskxr.png';
% % ctorig = 'example_0/inputct.nii.gz';
% % ctsegm = 'example_0/inputct.nii.gz_maskct.nii.gz';

xrorig = 'example_1/inputxrorig.dcm';
xrsegm = 'example_1/inputxr_uint8.png_maskxr.png';
ctorig = 'example_1/inputct.nii.gz';
ctsegm = 'example_1/inputct.nii.gz_maskct.nii.gz';

% % DrugResistorScript(xrorig, xrsegm, ctorig, ctsegm, 'test1');
DrugResistorScript(xrorig, xrsegm, ctorig, ctsegm, 'test2');
