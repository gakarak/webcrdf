function DrugResistor(xrpath, ctdir, sta, outfile)

ver = 'v0.3';
logmess('');
logmess(['DrugResistor ' ver ' started']);
if ~startMessage(ver, nargin)
    return
end

try
    controlInputs(xrpath, ctdir, sta, outfile);
    if ctdir(end) ~= '\'
        ctdir = [ctdir '\'];
    end

    logmess(['Reading DICOM X-ray image from "' xrpath '"'], 1);
    xr0 = double(dicomread(xrpath));
    logmess('Segmenting X-ray image', 1);
    xrsgm = xraysegmreg(xr0);
    logmess('Describing X-ray image', 1);
    xrdesc = xraydescribe(xr0, xrsgm);
    % xrdesc = dlmread('xray_data\xr_desc.08bins.txt');

    logmess(['Reading DICOM CT image from "' ctdir '"'], 1);
    ctconvert(ctdir);
    logmess('Segmenting CT image', 1);
    ctsegmthres;
    logmess('Describing CT image', 1);
    ctdesc = ctdescribe;
    % ctdesc = dlmread('ct_data\ct_segm_w1000_b.tcd');

%     xrdesc = advNormalize(xrdesc, [3 120 1]); %# commented in ver 0.2
    xrdesc = xrdesc / sum(xrdesc(:));
    ctdesc = advNormalize(ctdesc, [12 12 6 4 4 4 3]);

    status = str2double(sta);
    status(status ~= 2) = 1;

    logmess('Loading the trained classifiers', 1);
    rp = trainClassifiers;
    [svm, ~, ~, lprob] = rp.predict({ctdesc, xrdesc}, status);

    logmess('Prediction Results', 1);
    str = sprintf('PredictedDrugResistance=%i  Probability=%i%%', ...
        svm, round(lprob * 100));
    logmess(str, 1);
    logmess(['Writing to file "' outfile '"'], 1);
    fid = fopen(outfile, 'wt');
    str = sprintf('PredictedDrugResistance=%i\nProbability=%i%%\n', ...
        svm, round(lprob * 100));
    fprintf(fid, str);
    fclose('all');
catch err
    logmess(['ERROR: ' err.message], 1);
end

end

function go = startMessage(ver, nargin)
% Displays the starting information

if nargin == 0
    disp(['DrugResistor ' ver ' - a software for prediction of drug ']);
    disp('resistance status of tuberculosis patients');
    disp(' ');
else
    disp(['DrugResistor ' ver]);
    if nargin < 4
        disp('Too few input arguments (must be 4)');
        disp(' ');
    end
end
if nargin < 4
    disp('Usage:    DrugResistor.exe xray_path ct_dir status output_file');
    disp('          "xray_path" is the path to patient`s X-ray Dicom file');
    disp('          "ct_dir" is the path to a directory containing ');
    disp('              patient`s CT DICOM files (DICOM DIR)');
    disp('          "status" is patient`s status of treatment which indicates');
    disp('              if the patient had already been treated before');
    disp('              2 - treated before, 1 - not treated, 0 - unknown');
    disp('          "output_file" is the path to output file');
end

go = nargin > 0;

end

function controlInputs(xrpath, ctdir, sta, outfile)
% Checks the existance of files, directories, the correctness of input data

if ~exist(xrpath, 'file')
   throw(MException('VerifyInput:FileNotFound', ...
       'X-ray DICOM file not found'));
end

if ~exist(ctdir, 'dir')
   throw(MException('VerifyInput:PathNotFound', ...
       'CT DICOM directory not found'));
end

s = str2double(sta);
if s < 0 || s > 2
    throw(MException('VerifyInput:StatusOfTreatment', ...
       'Status of treatment indicator should be between 0 and 2'));
end

if exist(outfile, 'file')
    logmess(['The output file "' outfile '" will be overwritten'], 1);
end
   
end

function [moved, mask] = elastix_reg2D(fixed, moving, movmask)

delete('registration\result.bm*');
delete('registration\result.0.bm*');

imwrite(fixed, 'registration\fixed.bmp');
imwrite(moving, 'registration\moving.bmp');
imwrite(movmask, 'registration\mask_moving.bmp');

system(['registration\elastix -f registration\fixed.bmp ' ...
    '-m registration\moving.bmp -out registration ' ...
    '-p registration\parameters_BSpline.txt']);
changeTransformParametersFile('registration\TransformParameters.0.txt', ...
    'registration\TransformParameters.txt');
system(['registration\transformix -in registration\mask_moving.bmp ' ...
    '-out registration -tp registration\TransformParameters.txt']);

moved = imread('registration\result.0.bmp');
mask = imread('registration\result.bmp');

end

function b = advNormalize(a, dims)
% performes advanced descriptor matrix normalization so that for each
% distance (D) dimension sum of elements is eequal to 1 (normalization is
% performed separately for each distance D). Here 'a' is initial descriptor
% matrix (noe matrix == one row), and dims is vector of dimensions + the
% last element representing the number of distance (D) dimension.

ds = zeros(dims(1:end-1));
for i=1:numel(ds)
    [s1 s2 s3 s4 s5 s6 s7 s8 s9 s10] = ind2sub(dims(1:end-1), i);
    sb = [s1 s2 s3 s4 s5 s6 s7 s8 s9 s10];
    ds(i) = sb(dims(end));
end
ds = permute(ds, numel(dims)-1:-1:1);
ds = ds(:);

b = a * 0;
for i=1:dims(dims(end))
    b(:, ds == i) = scal(a(:, ds == i)', zeros(1, ...
        size(a(:, ds == i)', 2)), sum(a(:, ds == i), 2)')';
end

% useful for debug. explores cross-distance correlation of descriptor
% elements
% rw = b(1, :);
% figure, scatter(rw(ds == 1), rw(ds == 2))

end

function changeTransformParametersFile(oldfile, newfile)

f1 = fopen(oldfile, 'rt');
f2 = fopen(newfile, 'wt');

toFind = '(FinalBSplineInterpolationOrder';
while ~feof(f1)
    line = fgetl(f1);
    if length(line) > length(toFind) && ...
            strcmpi(toFind, line(1:length(toFind)))
        line = [toFind ' 0)'];
    end
    fprintf(f2, '%s\n', line);
end

fclose(f1);
fclose(f2);
fclose('all');

end

function ct = ctconvert(inputdir)

delete('ct_data\init\ct.*')
system(['ct_data\dcm2nii.exe ' inputdir]);

files = dir(inputdir);
for i = 1:length(files)
    nm = files(i).name;
    
    if (length(nm) > 4 && (strcmpi(nm(end - 3:end), '.img') ...
            || strcmpi(nm(end - 3:end), '.hdr')))
        movefile([inputdir nm], ['ct_data\init\ct' nm(end - 3:end)]);
    end
end

ct = analyze75read('ct_data\init\ct.hdr');

end

function desc = ctdescribe

delete('ct_data\ct_segm_w1000_b.*');
system('ct_data\Tiler3D.exe ct_data\ct_segm.hdr');
desc = dlmread('ct_data\ct_segm_w1000_b.tcd');

end

function ctsegmthres
% Performs segmentation of DICOM CT image using threshold-based method

delete('ct_data\ct_segm.*');
system('ct_data\Segmentation.exe ct_data\init');
movefile('ct_data\init\ct_segm.img', 'ct_data\ct_segm.img');
movefile('ct_data\init\ct_segm.hdr', 'ct_data\ct_segm.hdr');

end

function logmess(str, onScreen)

if nargin < 2
    onScreen = false;
end

logfile = 'DrugResistor.log';
fid = fopen(logfile, 'at');

if ~isempty(str)
    v = clock;
    fprintf(fid, '%i.%02i.%02i %02i:%02i:%02i - %s\n', ...
        v(1), v(2), v(3), v(4), v(5), floor(v(6)), str);
else
    fprintf(fid, '\n\n');
end
fclose('all');

if onScreen
    disp(['# ' str]);
end

end

function rp = trainClassifiers

ct = dlmread('training\training_data_ct.txt');
xr = dlmread('training\training_data_xray.txt');
sp = dlmread('training\training_data_suppl.txt');
dr = dlmread('training\training_data_dr.txt');
nm = dlmread('training\training_data_nmax.txt');

rp = ResistPredictor.fit({ct, xr}, sp, dr, nm);

end

function desc = xraydescribe(xray, sgm)
% Calculates X-ray image descriptor

%# ver 0.2
% rqs = quantile(xray(sgm(:)), [0.1 0.25 0.5 0.75]);
% i75 = quantile(xray(:), 0.75);
% b = [-0.2982 0.7903 0.7034 -0.3159 0.1149 32.2264];
% qc = [rqs i75 1] * b';

%# ver 0.3
xr = xray .* sgm;
qc = quantile(xr(xr(:) > 1), 0.5);

xr = xray - qc + 5000;
mnmx = [4534 5593];
xr = (xr - mnmx(1)) / (mnmx(2) - mnmx(1));
xr(xr < 0) = 0;
xr(xr > 1) = 1;
xr(sgm == 0) = 0;

imwrite(~sgm, 'xray_data\xr_segm.bmp');
imwrite(xr, 'xray_data\xr_equalized.bmp');

system(['xray_data\crdf2012-reduce-0.exe -images xray_data\list_bmp.txt ' ...
    '-masks xray_data\list_masks.txt -dt 2 -bn 8 -ds 1 3 5 -lu 0']);

desc = dlmread('xr_equalized.08.both.txt');
movefile('xr_equalized.08.both.txt', 'xray_data\xr_desc.08bins.txt');
movefile('xr_equalized.08bins.bmp', 'xray_data\xr_binned.08bins.bmp');

end

function sgm = xraysegmreg(xray)
% performs registration-based X-ray image segmentation with use of X-ray DB

xrcut = xray(401:end-400, 401:end-400);
q1 = quantile(xrcut(:), 0.01);
q2 = quantile(xrcut(:), 0.99);
xrsmall = 0.1 + 0.8 * (xray - q1) / (q2 - q1);
xrsmall = imresize(xrsmall, [256 256]);

rdn = [mean(xrsmall, 1) mean(xrsmall, 2)'];

tbl = dlmread('data\XRAY_ID_RHADON_256.txt');

nbest = 5;
rs = corr(rdn', tbl(:, 2:end)');
[~, idx] = sort(rs, 'descend');
bestIDs = tbl(idx(1:nbest), 1);
bestims = cell(1, nbest);
bestmsk = cell(1, nbest);
for i = 1:nbest
    bestims{i} = imread(sprintf('data\\xray_base\\%03i_256.bmp', bestIDs(i)));
    bestmsk{i} = imread(sprintf('data\\xray_base\\%03i_256.png', bestIDs(i)));
end

av = xrsmall * 0; 
for i = 1:nbest
    [~, sgm] = elastix_reg2D(xrsmall, bestims{i}, bestmsk{i});
    av = av + im2double(sgm) / nbest;
end
av = imresize(imfilter(av, fspecial('gaussian', [5 5], 1)), ...
            size(xray));
sgm = av > 0.5;
% imwrite(sgm, 'xray_data\xr_segm.png');

end