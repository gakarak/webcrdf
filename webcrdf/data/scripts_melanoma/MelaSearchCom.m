% % function MelaSearchCom(pathDB, filename, outfile, oiFlag)
function MelaSearchCom(pathDB, filename, oiFlag)

% initialize
warning off all

ver = 'v0.4(com)';
logmess('');
logmess(['MelaSearch ' ver ' started']);

outImages = false;
if nargin < 2 
    logmess('No input file specified!');
    
    fprintf(['\nUSAGE: MelaSearchCom.exe pathToDB inputFile.jpg [outputFile.txt]'...
        ' [-oi]\n\n']);
    return
end
% % elseif nargin < 3
% %     outfile = 'out.txt';
% % end
imDir = fileparts(filename);
outfile = [imDir, '/out.txt'];
if nargin >= 3
    outImages = strcmpi(oiFlag, '-oi');
end

% reading input
logmess(['Reading image file "' filename '"']);
handles.pathDB = pathDB;
handles.pathOut = imDir;
try
    im0 = imread(filename);
    if max(size(im0)) > 800 
        im0 = imresize(im0, 770 / size(im0, 2));
    end
    handles.data.im0 = im0;
    if length(size(handles.data.im0)) < 3
        throw(MException('VerifyInput:RGB', ...
             'Expected RGB image'));
    end
catch
    logmess(['Failed to read RGB image file "' filename '"']);
    fid = fopen([imDir,'/err.txt'], 'wt');
    fprintf(fid, ['ERROR: Failed to read RGB image file "' filename '"']);
    fclose(fid);
    return
end
handles.data.impath = filename;

if ~test_black(im0)
    logmess('Unsuitable image - black frame is too large');
    fid = fopen([imDir,'/err.txt'], 'wt');
    fprintf(fid, 'ERROR: Unsuitable image - black frame is too large');
    fclose(fid);
    return
end

% loading DB
handles = loadDB(handles);

logmess('Describing the image');
handles.data.desc = describeImage(handles.data.im0);

if sum(handles.db.dgn > 0) < 4
    messDB = 'The database is too small to perform searching';
    resDB = '---';
else
    logmess('Finding similar');
    [resDB, messDB, simImages] = findSimilar(handles);
end

logmess(messDB);

% Processin the image
logmess('Preprocessing image...');
handles = preprocessImage(handles);
logmess('Extracting features');
[resProc, handles] = extractFeatures(handles);
handles = makeView(handles);
procImage = fitImage(handles.proc.view);

fid = fopen(outfile, 'wt');
if resProc < 15
    strProc = '<span class=colorProbL>Not suspicious</span>';
elseif resProc < 25
    strProc = '<span class=colorProbM>Suspicious</span>';
else
    strProc = '<span class=colorProbH>Highly suspicious</span>';
end

fprintf(fid, ['Probability of Melanoma = %i%% (%s)\n' ...
    '%s\n%s\n'], resProc, strProc, messDB, handles.txtInfo);
fclose(fid);

if numel(outfile > 4)
    if strcmpi(outfile(end - 3:end), '.txt')
        outfile = outfile(1:end-4);
    end
end

if outImages
    imwrite(simImages{1}, [outfile '_sim1.png']);
    imwrite(simImages{2}, [outfile '_sim2.png']);
    imwrite(simImages{3}, [outfile '_sim3.png']);
    imwrite(simImages{4}, [outfile '_sim4.png']);
% %     txt=vision.TextInserter('1','FontSize',24,'Location',[100,100]);
% %     imwrite(step(txt,simImages{1}), [outfile '_sim1.png']);
% %     txt=vision.TextInserter('2','FontSize',24,'Location',[100,100]);
% %     imwrite(step(txt,simImages{2}), [outfile '_sim2.png']);
% %     txt=vision.TextInserter('3','FontSize',24,'Location',[100,100]);
% %     imwrite(step(txt,simImages{3}), [outfile '_sim3.png']);
% %     txt=vision.TextInserter('4','FontSize',24,'Location',[100,100]);
% %     imwrite(step(txt,simImages{4}), [outfile '_sim4.png']);
    imwrite(procImage, [outfile '_proc.png']);
end

%*****************************************************
%---- Specific functions
%*****************************************************

function [suitable, perc] = test_black(im)

gr = rgb2gray(im);

bw = gr < 20;
lbl = bwlabel(bw, 8);
blk = lbl * 0;

blk(lbl == lbl(1, 1)) = lbl(1, 1);
blk(lbl == lbl(1, end)) = lbl(1, end);
blk(lbl == lbl(end, 1)) = lbl(end, 1);
blk(lbl == lbl(end, end)) = lbl(end, end);
blk = blk > 0;

perc = sum(blk(:)) / numel(blk);
suitable = perc < 0.2;


function handles = preprocessImage(handles)

im = handles.data.im0;
logmess('Exstracting hair');
handles.proc.hair = hairmap(im);
hm = handles.proc.hair;
logmess('Exstracting reflections');
handles.proc.refl = reflections(im);
logmess('Segmenting via PCA-based method');
handles.proc.sgm1 = segmentPCA(im, hm);   % Segmentation via PCA

function [res, handles] = extractFeatures(handles)

im = handles.data.im0;
gr = im2double(rgb2gray(im));
hm = handles.proc.hair;
rfl = handles.proc.refl;

sgm = handles.proc.sgm1;

[~, ~, asym] = symmetries(sgm);  % Estimating Asymmetry
veil = blueveil(im, sgm);   % Detection of Blue-Whitish Veil
[nclrs, cls, hint] = colorsPresent(im, sgm, hm, rfl);   % Multiple Colors
rgr = regressarea(sgm);  % Detection of Regression Area
glb = globules(gr, sgm, hm, rfl);   % Finding Dots/Globules
stk = streaks(im, sgm);    % Detection of Streaks 
pnet = net(im, sgm);
brd = Border_dynamic(im, sgm);

ap = {'Absent', 'Present'};

str = [];
str = [str sprintf('Image size: %i x %i\n\n', size(im, 1), size(im, 2))];
if asym == 0
    str = [str sprintf('Asymmetry: no\n')];
elseif asym == 1
    str = [str sprintf('Asymmetry: along 1 axis\n')];
elseif asym == 2
    str = [str sprintf('Asymmetry: along 2 axes\n')];
end

str = [str sprintf('Border sharpness: %i / 8 \n',  brd)];

cols = [];
for ic = 1:length(cls)
    if cls(ic) == 1
        cols = [cols hint{ic} ', '];
    end
end
cols = cols(1:end - 2);
str = [str sprintf('Major colors: %s (%i)\n',  cols, nclrs)];

str = [str sprintf('Blue-whitish veil: %s\n',  ap{1 + veil})];
str = [str sprintf('Regression area: %s\n',  ap{1 + rgr})];
str = [str sprintf('Pigment network: %s\n',  ap{1 + pnet})];
str = [str sprintf('Dots & globules: %s\n',  ap{1 + glb})];
str = [str sprintf('Streaks: %s\n',  ap{1 + stk})];

handles.txtInfo = str;

% X = [asym veil rgr]; % v0.3
% B = [3.8063 -0.8717 -3.5443 -1.0153]';

% % X = [asym veil rgr stk cls(6)]; % v0.4, cls(6) == red
% % B = [4.1599 -0.8661 -2.7320 -1.1976 -1.6339 0.4837]';

X = [asym veil rgr stk]; % v0.4.3, cls(6) == red
B = [4.3544 -0.8549 -2.8819 -1.1724 -1.6800]';


logit_prob = mnrval(B, X);
logit_prob = logit_prob(2);

abcd = asym * 1.3 + 0.1 * brd + 0.5 * nclrs + ...
    0.5 * (glb + stk + rgr + pnet);

argenz = pnet * 2 + veil * 2 + stk + glb + rgr;

menz_neg = double(asym == 0) + double(nclrs == 1 && cls(4) == 0);
menz_pos = veil + glb + stk + double(nclrs > 4);
menzies = (menz_neg == 0) && (menz_pos > 0);

logmess(sprintf('ABCD = %.1f, 7-Point = %i, Menzies = %i (+%i, -%i)', ...
    abcd, argenz, menzies, menz_pos, menz_neg));
res = round(100 * logit_prob);
logmess(sprintf('Probability of melanoma = %i%%', res));

function [res, mess, simImages] = findSimilar(handles)

d = handles.data.desc;
descs = handles.db.descs;
dists = sum(abs(descs - repmat(d, size(descs, 1), 1)), 2);
ii = dists > 0.01 & handles.db.dgn > 0;
dgn = handles.db.dgn(ii);
% names = handles.db.names(ii);
dists = dists(ii);
[~, idx] = sort(dists, 'ascend');

thr = 4.21;
simImages = cell(1, 4);
for i = 1:4
    simImages{i} = fitImage(0.5);
end
for i = 1:min(4, sum(dists < thr))
    imsim = imread([handles.db.dir '/' handles.db.names{idx(i)}]);
    imsim(1:120, 1:120, :) = 0;
    if dgn(idx(i)) > 0
        imsim(1:120, 1:120, 4 - dgn(idx(i))) = 255;
    end
    simImages{i} = fitImage(imsim);
end

nmel = sum(dgn(idx(1:4)) == 3 & dists(idx(1:4)) < thr);
if sum(dists < thr) == 0
    mess = 'Not recognized as a dermatoscopy image';
    res = '--';
elseif min(4, sum(dists < thr)) < 3
    mess = 'Too few similar images found';
    res = '-';
elseif nmel == 0
    mess = 'Looks like a <span class=colorProbL>NEVUS</span> (no similar Melanoma images found)';
    res = num2str(nmel);
elseif nmel == 1
    mess = sprintf('Looks like a <span class=colorProbL>NEVUS</span> (%i similar Melanoma image found)', nmel);
    res = num2str(nmel);
elseif nmel < 4
    mess = sprintf('Looks like <span class=colorProbM>MELANOMA</span> (%i similar Melanoma images found)', nmel);
    res = num2str(nmel);
else
    mess = sprintf('Very similar to <span class=colorProbH>MELANOMA</span> (%i similar Melanoma images found)', nmel);
    res = num2str(nmel);
end

function handles = makeView(handles)

sgm1 = handles.proc.sgm1;

rgb = handles.data.im0;
rgb = colorRegion(rgb, ...
    handles.proc.refl | handles.proc.hair, [0 0.7 0.9] * 255);
rgb = colorRegion(rgb, bwmorph(sgm1, 'dilate', 4) & ~sgm1, [0 0 1] * 255);
handles.proc.view = rgb;

function logmess(str)

disp(str);

logfile = 'MelaSearch.log';
fid = fopen(logfile, 'at');

if ~isempty(str)
    v = clock;
    fprintf(fid, '%i.%02i.%02i %02i:%02i:%02i - %s\n', ...
        v(1), v(2), v(3), v(4), v(5), floor(v(6)), str);
else
    fprintf(fid, '\n\n');
end
fclose('all'); 

function im1 = fitImage(im0)

sz = [572 765 3];
if numel(im0) == 1
    im1 = imresize(im0, sz(1:2));
    return
end

sz0 = size(im0);
im1 = 0.5 * ones(sz);
if sz0(1) / sz0(2) > sz(1) / sz(2)
    im = imresize(im0, sz(1) / sz0(1));
    b = floor((sz(2) - size(im, 2)) / 2);
    im1(:, 1 + b:b + size(im, 2), :) = im2double(im);
else
    im = imresize(im0, sz(2) / sz0(2));
    b = floor((sz(1) - size(im, 1)) / 2);
    im1(1 + b:b + size(im, 1), :, :) = im2double(im);
end

function h = loadDB(handles)

logmess('Loading database...');
parDirDB = handles.pathDB;
parDBCSV = [parDirDB, '/MelaDB.csv'];
data = importdata(parDBCSV);

num=data.data;
% % [num str] = xlsread('MelaDB.xlsx');
handles.db.dir = parDirDB; %%str{1, 2};
handles.db.names = data.textdata; %%str(3:end, 1);
handles.db.dgn = data.data(:,1); %%num(:, 1);
handles.db.descs = data.data(:,2:end); %%num(:, 2:end);
handles.db.vol = size(data.data,1); %%size(num, 1);
logmess(sprintf('Databases with %i cases (%i malignant) loaded', ...
    handles.db.vol, sum(num(:, 1) == 3)));

% % % logmess('Loading database...');
% % % [num, str] = xlsread('MelaDB.xlsx');
% % % handles.db.dir = str{1, 2};
% % % handles.db.names = str(3:end, 1);
% % % handles.db.dgn = num(:, 1);
% % % handles.db.descs = num(:, 2:end);
% % % handles.db.vol = size(num, 1);
% % % logmess(sprintf('Databases with %i cases (%i malignant) loaded', ...
% % %     handles.db.vol, sum(num(:, 1) == 3)));

h = handles;

%*****************************************************
%---- Common functions
%*****************************************************

function desc = describeImage(im)

cmap = [0.21176	0.16471	0.16078;
        0.6549	0.49412	0.41961;
        0.88627	0.71765	0.63529;
        0.40784	0.3098	0.29412;
        0.82745	0.62353	0.51765;
        0.53333	0.40784	0.36863;
        0.98039	0.87451	0.85098;
        0.38039	0.21569	0.13725;
        0.70196	0.55294	0.52157;
        0.74118	0.62745	0.61176;
        0.52549	0.33333	0.25098;
        0.35686	0.25098	0.23137;
        0.65098	0.43137	0.33725;
        0.90196	0.75294	0.72941;
        0.78431	0.53333	0.4;
        0.83137	0.6549	0.61961];
      
comatrixParams = struct('irange', [1 size(cmap, 1)],...
    'grange', [0 3],...
    'ibins', size(cmap, 1),...
    'gbins', 4,...
    'dists', 1:5,...
    'abins', 4,...
    'LUT', []);
params = struct('descType', 'iid',...
    'comatrixParams', comatrixParams);
clear comatrixParams
params = calcLUT(params);

ind = rgb2ind(im, cmap) + 1;

cm = getdesc(ind, params);
for k = 1:size(cm, 3)
    cm(:, :, k) = cm(:, :, k) / sum(sum(cm(:, :, k)));
end
desc = cm(:)';

function params = calcLUT(params0)
% fills of params.comatrixParams.LUT data field

params = params0;
LUT = [];
dists = params.comatrixParams.dists;

j = 0;
for i=1:max(dists)
    d = round(sqrt(i^2 + j^2));
    if ~isempty(find(dists == d, 1))
        id = find(dists == d, 1);
        LUT = [LUT; dists(id), i, j];
    end
end

for i=-max(dists):max(dists)
    for j=1:max(dists)
        d = round(sqrt(i^2 + j^2));
        if ~isempty(find(dists == d, 1))
            id = find(dists == d, 1);
            LUT = [LUT; dists(id), i, j];
        end
    end
end

params.comatrixParams.LUT = LUT;

function hm = hairmap(gr)
% Hair detection on a dermatoscopic image.
% INPUT:    gr - Gray-Level image [0..255] uint8
% OUTPUT:   hm - Binary hairmap image

if length(size(gr)) > 2
    gr = rgb2gray(gr);
end
gr = im2double(gr);

fs1 = dirfilters(10, 6, 16);
fs2 = dirfilters(10, 0.5, 16);
r1 = dirfiltresponce(gr, fs1);
r2 = dirfiltresponce(gr, fs2);

hm = r1 - r2 > 0.08;
hm = bwmorph(hm, 'dilate', 1);

hm([1:10 end-9:end], :) = 0; 
hm(:, [1:10 end-9:end]) = 0;

function gss = dirfilters(sdx, sdy, N)

gs0 = fspecial('gaussian', [201 201], 40);
gsa = imresize(gs0, ceil([size(gs0, 1) * sdy / 40, ...
    size(gs0, 2) * sdx / 40]));
gsa = gsa / sum(gsa(:));
gss = cell(1, N);
gss{1} = gsa;

for i = 2:N
    gss{i} = imrotate(gsa, 180 / N * (i - 1));
    gss{i} = gss{i} / sum(sum(gss{i}));
end

function mresp = dirfiltresponce(gr, gss)

N = length(gss);
resps = zeros([size(gr), N]);
for i = 1:N
    rsp = imfilter(gr, gss{i});
    resps(:, :, i) = rsp;
end
mresp = min(resps, [], 3);

function rfl = reflections(gr)
% Detection of reflections on a dermatoscopic image.
% INPUT:    gr - Gray-Level image [0..255] uint8
% OUTPUT:   rfl - Reflections binary image

if length(size(gr)) > 2
    gr = rgb2gray(gr);
end
gr = im2double(gr);

fs1 = dirfilters(5, 5, 1);
r1 = dirfiltresponce(gr, fs1);
rfl = gr > 0.7 & (gr - r1) > 0.1;

rfl = bwmorph(rfl, 'dilate', 1);
rfl([1:10 end-9:end], :) = 0; 
rfl(:, [1:10 end-9:end]) = 0;

function segm = segmentPCA(im, hmap)
% Segmentation of lesion based on PCA and thresholding
% For correct segmentation the 'cameramask.png' file must be available
% INPUT:    im - RGB dermatoscopic image [0..255] uint8
%           hmap - Binary hairmap image
% OUTPUT:   segm - Extracted region of interest (lesion)

small = im;
gray0 = im2double(rgb2gray(small));

if exist('cameramask.png', 'file')
    cmask = imresize(imread('cameramask.png'), size(gray0), 'nearest');
else
    cmask = true(size(gray0));
end

r = 50;
small1 = small(1+r:end-r, r+1:end-r, :);
rc = small1(:, :, 1);
gc = small1(:, :, 2);
bc = small1(:, :, 3);
pc = princomp(double([rc(:) gc(:) bc(:)]));

rc = small(:, :, 1);
gc = small(:, :, 2);
bc = small(:, :, 3);
gray = double([rc(:) gc(:) bc(:)]) * pc(:, 1);


gray = reshape(gray, size(gray0));
gray = gray * sign(corr(gray(:), gray0(:)));
gray = (gray - min(gray(:))) / (max(gray(:)) - min(gray(:)));

remhair = gray;
remhair(hmap) = mean(gray(:));
remhair = imfilter(remhair, fspecial('gaussian', [31 31], 15));
remhair(~hmap) = gray(~hmap);
remhair = imfilter(remhair, fspecial('gaussian', [31 31], 15));
remhair(~hmap) = gray(~hmap);


blurred = imfilter(remhair, fspecial('gaussian', [31 31], 15));

blr = blurred(1+r:end-r, r+1:end-r);
thr = graythresh(blr) + 0.02;
binned = ~im2bw(blurred, thr);

binned = binned & cmask;
segm = biggestBinary(binned);

function sel = biggestBinary(bw, thres)

if nargin < 2
    [lbl, num] = bwlabel(bw);
    bestLbl = 0;
    maxSz = 0;
    for il = 1:num
        if maxSz < sum(lbl(:) == il)
            maxSz = sum(lbl(:) == il);
            bestLbl = il;
        end
    end
    sel = lbl == bestLbl;
else
    [lbl, num] = bwlabel(bw);
    sel = bw * 0;
    for il = 1:num
        if thres < sum(lbl(:) == il)
            sel(lbl(:) == il) = 1;
        end
    end
end

function segm = segmentFCM(im, hmap)
% Segmentation of lesion based on Fuzzy C-Means
% For correct segmentation the 'cameramask.png' file must be available
% INPUT:    im - RGB dermatoscopic image [0..255] uint8
%           hmap - Binary hairmap image
% OUTPUT:   segm - Extracted region of interest (lesion)

small = im;
gray0 = im2double(rgb2gray(small));

if exist('cameramask.png', 'file')
    cmask = imresize(imread('cameramask.png'), size(gray0), 'nearest');
else
    cmask = true(size(gray0));
end

rh = double(im) * 0;
for c = 1:3
    gray = im2double(im(:, :, c));
    remhair = gray;
    remhair(hmap) = mean(gray(:));
    remhair = imfilter(remhair, fspecial('gaussian', [31 31], 15));
    remhair(~hmap) = gray(~hmap);
    remhair = imfilter(remhair, fspecial('gaussian', [31 31], 15));
    remhair(~hmap) = gray(~hmap);
    
    rh(:, :, c) = remhair;
end

blurred = rh * 0;
for c = 1:3
    blr = imfilter(rh(:, :, c), ...
        fspecial('gaussian', [31 31], 15));
%     blr(~cmask) = mean(blr(:));
    blurred(:, :, c) = blr;
end

[cnt, u] = fcm(reshape(blurred, [numel(gray) 3]), 2, [2 20 1e-5 0]);
icnt = sum(cnt, 2) == min(sum(cnt, 2));
u = reshape(u(icnt, :), size(gray));
u(~cmask) = 0;
binned = im2bw(u, graythresh(u));

binned = binned & cmask;
segm = biggestBinary(binned);

function [segm, bw_edge]=edging(im)
% Input im - grayscaled or RGB dermatoscopic image [0..255]
% Output segm - nevus area
% bw_edge - nevus edge
sz=size(size(im)); 
if sz(2)==3 
    gray=rgb2gray(im);
else
    gray=im;
end
q=imadjust(gray, stretchlim(gray), [],2);
q2=im2bw(q);
BWs=edge(q2, 'sobel');
se90=strel('line', 3, 90);
se0=strel('line', 3, 0);
BWsdil=imdilate(BWs, [se90 se0]);
BWdfill=imfill(BWsdil, 'holes');
seD=strel('diamond', 1);
BWfinal=imerode(BWdfill, seD);
bww=bwareaopen(BWfinal,300);
se=strel('disk', 2);
bww2=imclose(bww, se);
[L, num]=bwlabel(bww2, 8);
if num>1
    feats=regionprops(L, 'all', 8);
    Areas=feats.Area;
    for ii=1:num
        Areas(ii)=feats(ii).Area;
    end
    q=sort(Areas, 'descend');
    bww2=bwareaopen(bww2,q(2)+10);
end
segm=imfill(bww2,'holes');
bw_edge=bwmorph(segm,'remove', inf);

function im = colorRegion(rgb, bw, clr)

im = uint8(rgb * 0);
for c = 1:3
    l = rgb(:, :, c);
    l(bw) = round(clr(c));
    im(:, :, c) = l;
end

function [sym1 sym2 asymm] = symmetries(segm)
% The function estimates lesion shape asymmetry.
% INPUT:    segm - Binary lesion segmentation
% OUTPUT:   sym1 - Symmetry rate along axis 1 [0..1]
%           sym2 - Symmetry rate along axis 2 [0..1]
%           asymm - Final Asymmetry [0,1,2]

[xx yy] = meshgrid(1:size(segm, 2), 1:size(segm, 1));

xc = sum(xx(:) .* segm(:)) / sum(segm(:));
yc = sum(yy(:) .* segm(:)) / sum(segm(:));

x = xx(:) - xc;
y = yy(:) - yc;
r2 = x.^2 + y.^2;
inert = zeros(2);
inert(1, 1) = sum((r2 - x .* x) .* segm(:));
inert(1, 2) = - sum((x .* y) .* segm(:));
inert(2, 1) = - sum((x .* y) .* segm(:));
inert(2, 2) = sum((r2 - y .* y) .* segm(:));

[a, ~] = eig(inert);
rotAngle = 180 / pi * atan(a(3) / a(1));

segm1 = double(segm);
segm1(ceil(yc - 1:yc + 1), ceil(xc - 1:xc + 1)) = 5;
segm1 = imrotate(segm1, rotAngle, 'bilinear'); 
[yc xc] = find(segm1 == max(segm1(:)), 1);

[n m] = size(segm1);
refl1 = segm1 * 0;
refl1(:, max(1, 2 * xc - m):min(2 * xc, m)) = ...
    segm1(:, min(2 * xc, m):-1:max(1, 2 * xc - m));
refl2 = segm1 * 0;
refl2(max(1, 2 * yc - n):min(2 * yc, n), :) = ...
    segm1(min(2 * yc, n):-1:max(1, 2 * yc - n), :);

sym1 = sum(segm1(:) > 0 & refl1(:) > 0) / sum(segm1(:) > 0);
sym2 = sum(segm1(:) > 0 & refl2(:) > 0) / sum(segm1(:) > 0);

asymm = sum([sym1 sym2] < 0.9);

function [stk, score] = streaks(im, segm)
% Identification of presence of Streaks
% INPUT:    im - RGB image [0..255] uint8
%           segm - Extracted region of interest, binary image
% OUTPUT:   stk - Presence of Streaks, binary feature (No/Yes) [0,1]
%           score - Floating-point score

HSV = rgb2hsv(im); 
V = HSV(:,:,3);
V_mean = mean(V(:)); % #25
V_std = std(V(:)); % #26
V_ratio = V_std / V_mean; % #27

gray = rgb2gray(im);
gray_segm = uint8(segm).*gray;

GLCM_segm = graycomatrix(gray_segm, 'Offset', [0 1; -1 1; -1 0; -1 -1], ...
    'Symmetric', true);
glcm_segm = sum(GLCM_segm, 3);
stats_GLCM = graycoprops(glcm_segm,'all'); % #28-31

x = [V_ratio stats_GLCM.Correlation 1];
b = [1.2780; -4.5601; 4.2857];
score = x * b;

thr = 0.19;
stk = double(score > thr);

function [rgr score area] = regressarea(segm)
% Identifies the presence of regression
% INPUT:    segm - Extracted region of interest, binary image
% OUTPUT:   rgr - Presence of regression, binary feature (No/Yes) [0,1]
%           score - Floating-point score of presence of regression [0..1]
%           area - Binary image displaying the regression area

conv = bwconvhull_alt(segm > 0);
[lbl, n] = bwlabel(conv & ~segm);
big = lbl == 1;
mx = sum(big(:));
for j = 2:n
    if sum(lbl(:) == j) > mx
        mx = sum(lbl(:) == j);
        big = lbl == j;
    end
end

area = big;
score = sum(big(:) > 0) / sum(segm(:) > 0);
thr = 0.052;
rgr = double(score > thr);

function P = bwconvhull_alt(BW)
% usage: P is a binary image wherein the convex hull of objects are returned, BW is the input binary image
% P= bwconvhull_alt(BW);

% warning off all
s=regionprops(logical(BW),'ConvexImage','BoundingBox');
P=zeros(size(BW));
for no=1:length(s)
P(s(no).BoundingBox(2):s(no).BoundingBox(2)+s(no).BoundingBox(4)-1,...
    s(no).BoundingBox:s(no).BoundingBox(1)+s(no).BoundingBox(3)-1)=s(no).ConvexImage;
end

function [glb, gmap, n] = globules(gr, segm, hairmap, refl)
% Identification of presence of Dots and Globules
% INPUT:    gr - Gray-Level image [0..255] uint8
%           segm - Extracted region of interest, binary image
%           hairmap - Binary hairmap image
%           refl - Binary reflections image
% OUTPUT:   glb - Presence of Dots/Globules, binary feature (No/Yes) [0,1]
%           gmap - Binary image displaying the detected structures
%           n - Number of Dots/Globules identified [0..1]

gr = im2double(gr);

fs1 = dirfilters(5, 5, 1);
fs2 = dirfilters(1.5, 1.5, 1);
fs3 = dirfilters(2.5, 2.5, 1);
fs4 = dirfilters(2, 2, 1);
r1 = dirfiltresponce(gr, fs1);
r2 = dirfiltresponce(gr, [fs2 fs3 fs4]);
sc = (r1 - r2) .* ~hairmap .* ~refl .* segm;

gmap = sc > 0.1;
[~, n] = bwlabel(gmap);
score = n / sum(segm(:));

thr = 0.0000051;
glb = score > thr;

function [nclrs clrs hint] = colorsPresent(im, segm, hmap, refl, thr)
% Identifies the presence of various colors on the image
% The 6 colors to be found: DarkBrown, LightBrown, White, Blue, Black, Red
% INPUT:    im - RGB dermatoscopic image [0..255] uint8
%           segm - Extracted region of interest (lesion)
%           hmap - Binary hairmap image
%           refl - Binary reflections image
%           thr - (optional) Threshold value (default 0.05)
% OUTPUT:   nclrs - Number of colors present [0:6]
%           clrs - Identifies which colors are present
%           hint - Returns the text hint (names of colors)

if nargin < 5
    thr = 0.13;
end
hint = {'DarkBrown', 'LightBrown', 'White', 'Blue', 'Black', 'Red'};

cmap = [0.37647	0.24314	0.20392;
    0.7098	0.43922	0.32941;
    0.85882	0.61176	0.59804;
    0.42451	0.34118	0.3598;
    0.20529	0.14902	0.12157;
    0.9098	0.37022	0.29804];

ind = rgb2ind(im, cmap);
ind(~segm) = size(cmap, 1) + 1;
ind(hmap) = size(cmap, 1) + 1;
ind(refl) = size(cmap, 1) + 1;
clrs = zeros(1, size(cmap, 1));
for i = 1:size(cmap, 1)
    clrs(i) = (sum(ind(:) == i - 1) / sum(segm(:))) > thr;
end

nclrs = sum(clrs);

function [veil, score] = blueveil(im, segm)
% Identification of presence of a Blue-Whitish veil
% INPUT:    im - RGB dermatoscopic image [0..255]
%           segm - Extracted region of interest, binary image
% OUTPUT:   veil - Presence of veil, binary feature (No/Yes) [0,1]
%           score - Floating-point score of presence of veil [0..1]

hsv1 = rgb2hsv(im);
sel = hsv1(:, :, 1) > 210 / 255;
sel = sel & hsv1(:, :, 1) < 240 / 255;
sel = sel & hsv1(:, :, 2) > 40 / 255;
sel = sel & segm;

score = sum(sel(:)) / sum(segm(:));
veil = double(score > 0.057);

function [network, Density, cell_mask]=net(im,segm,log_sigm)
% v2.0
% Input im - RGB dermatoscopic image [0..255]
%       segm - nevus area
%       log_sigm - sigma in the LOG filer for cell detection
% Output cell_mask - pigment network cell mask 
%       Density - graph density
%       network - presense of pigment network, binary value
if nargin < 3
    log_sigm=1.5;
end
gray=rgb2gray(im);
% [a,b]=size(gray);
edge_log=edge(gray,'log',[], log_sigm);
edge_open=bwareaopen(edge_log,10);

cell_mask = imfill(edge_open,'holes');
cell_mask=logical(cell_mask-edge_open);
cell_mask=bwareaopen(cell_mask, 10); % cell mask

L2=bwlabel(cell_mask,8);
Prop=regionprops(cell_mask,'Orientation','Area','Perimeter',...
    'EquivDiameter','Centroid');
LeP=max(max(L2));
Area=zeros(1,LeP); Orient=Area;
Perim=Area; Ediam=Area;
Centx=Area; Centy=Area;
for i=1:LeP
    Centx(i)=Prop(i).Centroid(1);
    Centy(i)=Prop(i).Centroid(2);
    Orient(i)=Prop(i).Orientation;
     Perim(i)=Prop(i).Perimeter;
    Area(i)=Prop(i).Area;
    Ediam(i)=Prop(i).EquivDiameter;
end

MDT=3*(sum(Ediam))/length(Ediam);
aaa=rot90(fliplr(Centx));
bbb=rot90(fliplr(Centy));
Koord=[aaa, bbb];
Rast=pdist(Koord);
Sf=squareform(Rast);
[s1,s2]=size(Sf);
rebr=zeros(s1,s2);
rebr(Sf<=2*MDT)=1;
rebr=rebr-diag(diag(rebr));
Prop_bord=regionprops(segm,'Area');
LeP_bord=length(Prop_bord);
Area_bord=zeros(1,LeP_bord);
for i=1:LeP_bord
    Area_bord(i)=Prop_bord(i).Area;
end
E=sum(sum(rebr));
Density=E/(LeP*log(max(Area_bord)));
network = Density > 0.31;

function [nsharp, entropy, bord_dyn] = Border_dynamic(im,segm)
% v2.0
% Input im - grayscaled or RGB dermatoscopic image [0..255]
% segm - nevus area
% Output bord_dyn - array containing numbers of pixel with the intensity
% which lay between adjacent sections in the sectors
sz=size(size(im)); 
if sz(2)==3 
    gray=rgb2gray(im);
else
    gray=im;
end
height_step = 10;
angle_step = 45;
[a, b]=size(gray);

f=fspecial('average',10);
I_filtered=imfilter(gray,f,'replicate');
Ifd=double(I_filtered);
UN=double(ones(a,b)*255);
mesh_im=UN-Ifd;
IM=uint8(mesh_im);
s=0:height_step:250;
Volume=sum(sum(IM));

LeS=length(s);
Squa=zeros(1,LeS);
jk=360/angle_step;
[sector_masks] = sectors(angle_step, segm);
bord_dyn=zeros(jk,LeS-1);
for j=1:jk
    xd=uint8(Ifd).*uint8(sector_masks(:,:,j));
    for i=1:LeS-1
        q=find(xd>=s(i)&xd<s(i+1));
        Squa(j,i)=length(q);
        bord_dyn(j,i)=(Squa(j,i)*s(i))/Volume;
    end
end
bord_dyn(:,1)=[];
b = bord_dyn';
b = scal(b, sum(b) * 0, sum(b));
b = b + 1e-8;
entropy = - sum(b .* log2(b)) / log2(24);
nsharp = sum(entropy < 0.45);

function [sector_masks] = sectors(angle_step, segm)

% Input im - grayscaled or RGB dermatoscopic image [0..255]
% angle_limit - angle step for declaration number of sectors
% Output sector_masks - [a,b,n] array.
% n - number of sectors produced after execution
% one can use next expression for understanding
% imshow(sector_masks(:,:,1))

bord_full = segm;
Prop_bord=regionprops(bord_full,'Area','Centroid','PixelList');
LeP_bord=length(Prop_bord);
Area_bord=zeros(1,LeP_bord);
Centx=zeros(1,LeP_bord);
Centy=zeros(1,LeP_bord);
for i=1:LeP_bord
    Centx(i)=Prop_bord(i).Centroid(1);
    Centy(i)=Prop_bord(i).Centroid(2);
    Area_bord(i)=Prop_bord(i).Area;
end
Pixels=Prop_bord(1).PixelList;
XX=Pixels(:,1);
YY=Pixels(:,2);
XXX=XX-Centx(1);
YYY=YY-Centy(1);
[theta2, rho]=cart2pol(YYY,XXX);
theta=theta2+pi;
s=0:angle_step:360;
s=s*pi/180;
[aa,bb]=size(bord_full);
sector_masks=zeros(aa,bb,length(s)-1);
for i=1:length(s)-1
    ind=find(theta>=s(i)&theta<=s(i+1));
    [Y2,X2] = pol2cart(theta2(ind),rho(ind));
    kk1=X2+Centx(1); kk1=int16(kk1);
    kk2=Y2+Centy(1); kk2=int16(kk2); 
    for j=1:length(kk1)
        sector_masks(kk2(j),kk1(j),i)=bord_full(kk2(j),kk1(j));
    end
end

function [desc imb] = getdesc(im, params)
% returns descriptor - a vector of numbers (v2)

im = single(im);
gy = imfilter(im, -fspecial('sobel'));
gx = imfilter(im, -fspecial('sobel')');
[gr ga] = cart2pol(gx, gy);
ga(ga < 0) = ga(ga < 0) + 2 * pi;
ga = ga / 2 / pi;
cp = params.comatrixParams;

switch params.descType
    case 'sta'
        desc = STAdesc(im, gr, ga);
        
    case 'iid'
        [desc imb] = IIDdesc(im, cp);
        
    case 'ggd'
        desc = GGDdesc(gr, cp);
        
    case 'iidgg'
        desc = IIDGGdesc(im, gr, cp);
        
    case 'iidgga'
        desc = IIDGGAdesc(im, gr, ga, cp);
        
    case 'ggda'
        desc = GGDAdesc(im, gr, ga, cp);
        
    otherwise
        throw(MException('DESC:Unknown', ['descType "' ...
            params.descType '" not implemented']));
end

desc = single(desc);

function desc = STAdesc(im, gr, ga)
% for 'sta' descType

desc = [mean(im(:)) std(im(:)) skewness(im(:)) ...
    mean(gr(:)) std(gr(:)) skewness(gr(:)) ...
    mean(ga(:)) std(ga(:)) skewness(ga(:))];

function [desc imb] = IIDdesc(im, cp)
% for 'iid' descType

% desc = [];
desc = single(zeros(cp.ibins, cp.ibins, length(cp.dists)));
for id=1:length(cp.dists)
    il = cp.LUT(:, 1) == cp.dists(id);
    off = cp.LUT(il, :);
    off = off(:, 2:3);
    [cm imb] = graycomatrix(im, 'Offset', off, 'Symmetric', true,...
        'GrayLimits', cp.irange, 'NumLevels', cp.ibins);
%     desc = [desc sum(cm, 3)];
    desc(:, :, id) = sum(cm, 3);
end
% desc = desc(:) / sum(desc(:));

function desc = GGDdesc(gr, cp)
% for 'ggd' descType

desc = [];
for id=1:length(cp.dists)
    il = cp.LUT(:, 1) == cp.dists(id);
    off = cp.LUT(il, :);
    off = off(:, 2:3);
    cm = graycomatrix(gr, 'Offset', off, 'Symmetric', true,...
        'GrayLimits', cp.grange, 'NumLevels', cp.gbins);
    desc = [desc sum(cm, 3)];
end
desc = desc(:) / sum(desc(:));

function desc = IIDGGdesc(im, gr, cp)
% for 'iidgg' descType

desc = [];
[m imbins] = graycomatrix(im, 'GrayLimits', ...
    cp.irange, 'NumLevels', cp.ibins);
[m grbins] = graycomatrix(gr, 'GrayLimits', ...
    cp.grange, 'NumLevels', cp.gbins);

[i j] = meshgrid(1:size(im, 1), 1:size(im, 2));
i = i(:);
j = j(:);

for id=1:length(cp.dists)
    il = cp.LUT(:, 1) == cp.dists(id);
    offs = cp.LUT(il, :);
    offs = offs(:, 2:3);
    
    cm = zeros([cp.ibins cp.ibins cp.gbins cp.gbins]);
    for io=1:size(offs, 1)
        offset = offs(io, :);
        i2 = i + offset(1);
        j2 = j + offset(2);
        outsideBounds = find(j2 < 1 | j2 > size(im, 2) ...
            | i2 < 1 | i2 > size(im, 1));
        i2(outsideBounds) = [];
        j2(outsideBounds) = [];

        v1i = shiftdim(imbins,1);
        v1i = v1i(:);
        v1i(outsideBounds) = [];
        v1g = shiftdim(grbins,1);
        v1g = v1g(:);
        v1g(outsideBounds) = [];

        Index = i2 + (j2 - 1)*size(im, 1);
        v2i = imbins(Index);
        v2g = grbins(Index);

        Ind = [v1i v2i v1g v2g];
        oneGLCM = accumarray(Ind, 1, ...
            [cp.ibins cp.ibins cp.gbins cp.gbins]);

        cm = cm + oneGLCM;
    end
    cm = cm + permute(cm, [2 1 3 4]);
    cm = cm + permute(cm, [1 2 4 3]);

    desc = [desc cm(:)];
end
desc = desc(:) / sum(desc(:));

function desc = IIDGGAdesc(im, gr, ga, cp)
% for 'iidgga' descType

desc = [];
[m imbins] = graycomatrix(im, 'GrayLimits', cp.irange, 'NumLevels', cp.ibins);
[m grbins] = graycomatrix(gr, 'GrayLimits', cp.grange, 'NumLevels', cp.gbins);

[i j] = meshgrid(1:size(im, 1), 1:size(im, 2));
i = i(:);
j = j(:);

for id=1:length(cp.dists)
    il = cp.LUT(:, 1) == cp.dists(id);
    offs = cp.LUT(il, :);
    offs = offs(:, 2:3);
    
    cm = zeros([cp.ibins cp.ibins cp.gbins cp.gbins cp.abins]);
    for io=1:size(offs, 1)
        offset = offs(io, :);
        i2 = i + offset(1);
        j2 = j + offset(2);
        outsideBounds = find(j2 < 1 | j2 > size(im, 2) | i2 < 1 | i2 > size(im, 1));
        i2(outsideBounds) = [];
        j2(outsideBounds) = [];

        v1i = shiftdim(imbins,1);
        v1i = v1i(:);
        v1i(outsideBounds) = [];
        v1g = shiftdim(grbins,1);
        v1g = v1g(:);
        v1g(outsideBounds) = [];
        a1 = shiftdim(ga,1);
        a1 = a1(:);
        a1(outsideBounds) = [];

        Index = i2 + (j2 - 1)*size(im, 1);
        v2i = imbins(Index);
        v2g = grbins(Index);
        a2 = ga(Index);

        a = abs(a1 - a2);   % 0..2*pi
        a(a > pi) = 2 * pi - a(a > pi); % 0..pi
        a = 1 + floor(a / pi * cp.abins);
        a(a > cp.abins) = cp.abins;
        
        Ind = [v1i v2i v1g v2g a];
        oneGLCM = accumarray(Ind, 1, [cp.ibins cp.ibins cp.gbins cp.gbins cp.abins]);

        cm = cm + oneGLCM;
    end
    cm = cm + permute(cm, [2 1 3 4 5]);
    cm = cm + permute(cm, [1 2 4 3 5]);

    desc = [desc cm(:)];
end
desc = desc(:) / sum(desc(:));

function desc = IIDGGAdesc_obsolete(im, gr, ga, cp)
% for 'iidgg' descType. Function is obsolete, 'cause it's very slow. But
% is compared to the new version and the results matched

desc = [];
[m imbins] = graycomatrix(im, 'GrayLimits', cp.irange, 'NumLevels', cp.ibins);
[m grbins] = graycomatrix(gr, 'GrayLimits', cp.grange, 'NumLevels', cp.gbins);

iidgga = zeros(cp.ibins, cp.ibins, length(cp.dists), cp.gbins, cp.gbins, cp.abins);
for i=1:size(im, 1)
    for j=1:size(im, 2)
        for il=1:size(cp.LUT, 1)
            it = cp.LUT(il, 2);
            jt = cp.LUT(il, 3);
            if (i + it <= size(im, 1)) && (j + jt <= size(im, 2))...
                    && (i + it > 0) && (j + jt > 0)
                i1 = imbins(i, j);
                i2 = imbins(i + it, j + jt);
                d = cp.LUT(il, 1);
                g1 = grbins(i, j);
                g2 = grbins(i + it, j + jt);
                a = abs(ga(i, j) - ga(i + it, j + jt));
                if a > pi
                    a = 2-pi - a;
                end
                a = 1 + floor(a / pi * cp.abins);
                a = min(a, cp.abins);
                
                iidgga(i1, i2, d, g1, g2, a) = iidgga(i1, i2, d, g1, g2, a) + 1;
            end
        end
    end
end
iidgga = iidgga + permute(iidgga, [2 1 3 4 5 6]);
iidgga = iidgga + permute(iidgga, [1 2 3 5 4 6]);
desc = iidgga;
desc = desc(:) / sum(desc(:));

function desc = GGDAdesc(im, gr, ga, cp)
% for 'ggda' descType

desc = [];
[m grbins] = graycomatrix(gr, 'GrayLimits', cp.grange, 'NumLevels', cp.gbins);

[i j] = meshgrid(1:size(im, 1), 1:size(im, 2));
i = i(:);
j = j(:);

for id=1:length(cp.dists)
    il = cp.LUT(:, 1) == cp.dists(id);
    offs = cp.LUT(il, :);
    offs = offs(:, 2:3);
    
    cm = zeros([cp.gbins cp.gbins cp.abins]);
    for io=1:size(offs, 1)
        offset = offs(io, :);
        i2 = i + offset(1);
        j2 = j + offset(2);
        outsideBounds = find(j2 < 1 | j2 > size(im, 2) | i2 < 1 | i2 > size(im, 1));
        i2(outsideBounds) = [];
        j2(outsideBounds) = [];

        v1g = shiftdim(grbins,1);
        v1g = v1g(:);
        v1g(outsideBounds) = [];
        a1 = shiftdim(ga,1);
        a1 = a1(:);
        a1(outsideBounds) = [];

        Index = i2 + (j2 - 1)*size(im, 1);
        v2g = grbins(Index);
        a2 = ga(Index);

        a = abs(a1 - a2);   % 0..2*pi
        a(a > pi) = 2 * pi - a(a > pi); % 0..pi
        a = 1 + floor(a / pi * cp.abins);
        a(a > cp.abins) = cp.abins;
        
        Ind = [v1g v2g a];
        oneGLCM = accumarray(Ind, 1, [cp.gbins cp.gbins cp.abins]);

        cm = cm + oneGLCM;
    end
    cm = cm + permute(cm, [1 3 2]);

    desc = [desc cm(:)];
end
desc = desc(:) / sum(desc(:));