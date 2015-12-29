function [slwnd, slwndc, spwise, spwisec, cbir, cbirc] = genCTpreviews(im3, sgm, z2xy)

tic
cbir = [];
cbirc = [];
try
    [cbir, cbirc] = genCBIRpreview(im3, sgm, z2xy);
catch e
    disp('Failed to generate CBIR-based preview');
    disp(e.message);
end
toc

tic
spwise = [];
spwisec = [];
try
    [spwise, spwisec] = genSPWISEpreview(im3, sgm, z2xy);
catch e
    disp('Failed to generate SP-wise preview');
    disp(e.message);
end
toc

tic
slwnd = [];
slwndc = [];
try
    [slwnd, slwndc] = genSLWNDpreview(im3, sgm, z2xy);
catch e
    disp('Failed to generate Sliding-Window preview');
    disp(e.message);
end
toc

function [cbir, cbirc] = genCBIRpreview(im3, sgm, z2xy)

im = single(im3);
im = im / 1500;
im(im > 1) = 1;
im(im < 0) = 0;
msk = sgm > -2000;

[kk, kid] = candidateslices(msk);
selslices = im(:, :, kk);

im0 = selslices;
msk1 = msk(:, :, kk);
im1 = im0;
im1(~msk1) = nan;

cls = dlmread('refslices_cl_IIID[12,12,5].txt');
cls = advNormalize(cls, 5);

htmp = heatmapRCBIR(cls, im1);
hsv = preparehsv((htmp / 1.5) - 0.5, im1);
maxheat = maxheatslices(htmp, kid);

selslices = selslices(:, :, maxheat);
hsv = hsv(:, :, maxheat, :);

[cbir, cbirc] = composepreview(selslices, im, z2xy, hsv, maxheat, kk);

function [spwise, spwisec] = genSPWISEpreview(im3, sgm, z2xy)

im = single(im3);
im = im / 1500;
im(im > 1) = 1;
im(im < 0) = 0;
msk = sgm > -2000;

[kk, kid] = candidateslices(msk);
selslices = im(:, :, kk);

im1 = selslices;
msk1 = msk(:, :, kk);
im1(~msk1) = nan;

p = '16_0.3';
n = '8';
lib = dlmread(['tbcl_SP_' p '_lib_' n '.txt']);
descs = dlmread(['tbcl_SPCM_' p '_lib' n '.txt']);
cls = dlmread('tbcl_Clear_Tubercul.txt');

scores = spscores(descs, cls);
ss = strsplit(p, '_');

[~, ~, ~, ~, ftMaps] = superdescribe(im1, str2double(ss{1}), ...
    str2double(ss{1}), [], lib);

htmp = heatmapCM(scores, ftMaps);
hsv = preparehsv(htmp * 0.7, im1);
maxheat = maxheatslices(htmp, kid);

selslices = selslices(:, :, maxheat);
hsv = hsv(:, :, maxheat, :);

[spwise, spwisec] = composepreview(selslices, im, z2xy, hsv, maxheat, kk);

function [slwnd, slwndc] = genSLWNDpreview(im3, sgm, z2xy)

im = single(im3);
im = im / 1500;
im(im > 1) = 1;
im(im < 0) = 0;
msk = sgm > -2000;

[kk, kid] = candidateslices(msk);
selslices = im(:, :, kk);

p = '16_0.3';
n = '8';
ss = strsplit(p, '_');

lib = dlmread(['tbcl_SP_' p '_lib_' n '.txt']);
descs = dlmread(['tbcl_SPCM_' p '_lib' n '.txt']);
cls = dlmread('tbcl_Clear_Tubercul.txt');

[~, mn, pc, allB] = spscores(descs, cls);

im1 = selslices;
msk1 = msk(:, :, kk);

htmp = heatmapSW(im1, msk1, mn, pc, allB, str2double(ss{1}), str2double(ss{1}), lib);
im1(~msk1) = nan;
hsv = preparehsv(htmp - 0.5, im1);
maxheat = maxheatslices(htmp, kid);

selslices = selslices(:, :, maxheat);
hsv = hsv(:, :, maxheat, :);

[slwnd, slwndc] = composepreview(selslices, im, z2xy, hsv, maxheat, kk);