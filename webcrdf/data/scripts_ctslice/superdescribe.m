function [fhists, chist, ccm, spFeatures, ftMaps] = superdescribe(im3, sz, reg, resz, dct, msk3)
%v3.4
%[fhists, chist, ccm, spFeatures, ftMaps] = 
%= SUPERDESCRIBE(im3, sz, reg, resz, dct, msk3)
%
%INPUT:
%im3 - input 2D or 3D image;
%sz - superpixel grid size;
%reg - regularizer;
%resz - then image is resized to max(size(im))=resz before
%generating superpizels (specify [] to omit resizing);
%dct - (optional or []) pre-calculated superpixel dictionary;
%msk3 - (optional) input image mask, if not specified automatically
%generated as msk3 = ~isnan(im3);
%
%OUTPUT:
%fhists - historgams of 6 superixel features (mean, sd, entropy, mean
%gradient magnitude, "eccentricity" and squareness);
%chist - histogram of superpixels classes accoring to the dictionary 'dct'
%specified;
%ccm - co-occurrence matrix of superpixels classes accoring to the 
%dictionary 'dct' specified;
%spFeatures - Nx6 table contains features ofr aech superpixel;
%ftMap - 8-component cell structure, containing superpixel features (6), 
%superpixel number slice-wise (1) and superpixel dictionary class (1) 
%mapped to the original image;

if isempty(resz)
    resz = 0;
end

if nargin < 5
    dct = [];
end

if nargin < 6
    msk3 = ~isnan(im3);
end

m = 8;  % hist size

ft = {'mn', 'sd', 'en', 'gr', 'ec', 'sq'};
mnmx = {[0 1], [0 0.4], [0 1], [0 4], [3.54 7.54], [0 1]};

if nargout > 4
    z = single(im3 * 0);
    ftMaps = {z, z, z, z, z, z, z, z};
end

hists = cell(1, length(ft));
for j = 1:length(ft)
    hists{j} = zeros(1, 8);
end

if ~isempty(dct)
    ncl = size(dct, 1) - 1;
    chist = zeros(1, ncl);
    ccm = zeros(ncl);
else
    ncl = 0;
    chist = [];
    ccm = [];
end

spFeatures = [];

for k = 1:size(im3, 3)
    if size(im3, 3) > 1
        disp(num2str(k));
    end
    im = single(im3(:, :, k));
    msk = msk3(:, :, k);

    if sum(msk(:)) > 100
        jj = max(1, find(sum(msk) > 3, 1, 'first') - 16)...
            :min(size(im, 2), find(sum(msk) > 0, 1, 'last') + 16);
        ii = max(1, find(sum(msk, 2) > 3, 1, 'first') - 16)...
            :min(size(im, 1), find(sum(msk, 2) > 0, 1, 'last') + 16);
        
        msk = msk(ii, jj);
        im = im(ii, jj);
        
        if resz > 0
            r = resz / max(size(im));
            small = imresize(im, r, 'nearest');
        else
            small = im;
        end
            
        spx = my_slic(small, sz, reg);
        spx = imresize(spx, size(im), 'nearest');

    %     {'mean', 'std', 'eccentr', 'entropy'};
        [fts, cls, mps] = calcfeatures(im, msk, spx, dct);
        
        for j = 1:length(ft)
            l = mnmx{j};
            hists{j} = hists{j} + hist(fts(:, j), xhist(l(1), l(2), m));
        end
        if ~isempty(dct)
            chist = chist + hist(cls, xhist(1, ncl, ncl));
            cm = spcooccur(mps{7} + 1, mps{8}, ncl);
            ccm = ccm + cm;
        end
        if nargout > 4
            for j = 1:length(mps)
                z = ftMaps{j};
                z(ii, jj, k) = mps{j};
                ftMaps{j} = z;
            end
        end
        
        spFeatures = [spFeatures; fts];
        
%         sm = imresize(msk, size(small), 'nearest');
%         mn = imresize(mn, size(small), 'nearest');
%         sd = imresize(sd, size(small), 'nearest');
%         ec = imresize(ec, size(small), 'nearest');
%         en = imresize(en, size(small), 'nearest');
% 
%         saveslice(small, sm, mn, sprintf('sp_reg%.3f_mean_%03i.png', p, k));
%         saveslice(small, sm, sd / 0.4, sprintf('sp_reg%.3f_std_%03i.png', p, k));
%         saveslice(small, sm, (ec - 3.54) / 4, sprintf('sp_reg%.3f_ecc_%03i.png', p, k));
%         saveslice(small, sm, en, sprintf('sp_reg%.3f_ent_%03i.png', p, k));
    end
end

fhists = [];
for j = 1:length(ft)
    fhists = [fhists hists{j}];
end

end

function cm = spcooccur(spmap, clmap, ncl)

cls = zeros(1, max(spmap(:)) - min(spmap(:)) + 1);
for i = 1:max(spmap(:))
    s = spmap == i;
    cls(i) = mean(clmap(s(:)));
end

cm = zeros(ncl);

for i = min(spmap(:)):max(spmap(:))
    s = spmap == i;
    cl1 = mean(clmap(s(:)));
    
    if cl1 > 0
        s = bwmorph(s, 'dilate', 1) & ~s;
        s = s & spmap > i;
        sp2s = unique(spmap(s(:)));
        cl2s = cls(sp2s);
        cl2s = cl2s(cl2s > 0);
        cm(cl1, cl2s) = cm(cl1, cl2s) + 1;
    end
end
cm = cm + cm';

end

function saveslice(im, msk, ft, name)

ft(ft < 0.05) = 0.05;
ft(ft > 0.95) = 0.95;
im(im < 0.05) = 0.05;
im(im > 0.95) = 0.95;

hsv = ones(size(msk, 1), size(msk, 2), 3);
hsv(:, :, 1) = 0.7 - 0.7 * ft;
hsv(:, :, 3) = 0.3 + 0.7 * im;
rgb = hsv2rgb(hsv);
imwrite(rgb, name);

end

function [fts, cls, mps] = calcfeatures(im, msk, sgm, dct)

if ~isempty(dct)
    sdev = dct(1, :);
    dct = dct(2:end, :);
end

% ft = {'mn', 'sd', 'en', 'gr', 'ec', 'sq'};
fts = zeros(max(sgm(:)), 6);
cls = zeros(max(sgm(:)), 1);

gx = imfilter(im, -fspecial('sobel')');
gy = imfilter(im, -fspecial('sobel'));
[~, gm] = cart2pol(gx, gy);
gm([1 end], :) = 0;
gm(:, [1 end]) = 0;

mn = im * 0;
sd = im * 0;
en = im * 0;
gr = im * 0;
ec = im * 0;
sq = im * 0;
cl = im * 0;
[jj, ii] = meshgrid(1:size(im, 2), 1:size(im, 1));
an = ~isnan(im);
j = 0;
for i = min(sgm(:)):max(sgm(:))
    spx = sgm == i;
    inner = bwmorph(spx, 'erode');
    m = sum(inner(:));
    if m > 4
        if sum(msk(spx(:)) == 0) == 0
            j = j + 1;
            
            ec0 = sum(spx(:) & ~inner(:)) / sqrt(sum(spx(:)));
            if sum(inner(:)) < 4
                inner = spx;
            end
            mn0 = mean(im(inner(:) & an(:)));
            sd0 = std(im(inner(:) & an(:)));
            gr0 = mean(gm(inner(:) & an(:)));
            n = 8;
            h = hist(im(inner(:) & an(:)), 0.5 * 1/n + 1/n * (0:n - 1));
            h = h(h > 0) / sum(h);
            en0 = - sum(h .* log(h)) / log(n);
            
            bb = spx(min(ii(spx(:))):max(ii(spx(:))), ...
                min(jj(spx(:))):max(jj(spx(:))));
            sq0 = sum(bb(:)) / numel(bb);
            
            if ~isempty(dct)
                ds = pdist2(double([mn0 sd0 en0 gr0 ec0 sq0]) ./ sdev, dct);
                cl0 = find(ds == min(ds), 1);
            else
                cl0 = 0;
            end
            
            fts(j, :) = [mn0 sd0 en0 gr0 ec0 sq0];
            cls(j) = cl0;
            mn(spx) = mn0;
            sd(spx) = sd0;
            en(spx) = en0;
            gr(spx) = gr0;
            ec(spx) = ec0;
            sq(spx) = sq0;
            cl(spx) = cl0;
        end
    end
end

mps = {mn, sd, en, gr, ec, sq, sgm, cl};

end

function x = xhist(mn, mx, n)

stp = (mx - mn) / n;
x = mn + 0.5 * stp + stp * (0:n - 1);

end

function sp = my_slic(im, sz, reg)

% sp = vl_slic(im, sz, reg);
num = round(size(im, 1) / sz) * round(size(im, 2) / sz);
[sp, ~] = slicmex(uint8(im), num, reg / 256);

end