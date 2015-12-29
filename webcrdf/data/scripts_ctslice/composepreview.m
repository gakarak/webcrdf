function [big, bigc] = composepreview(selslices, im3, z2xy, hsv, maxheat, kk)

frontal = zeros(size(im3, 1), size(im3, 2), 3) + 0.5;
sz = size(frontal);
fr = im3(end/2, :, :);
fr = reshape(fr, size(im3, 1), size(im3, 3))';
frc = zeros([size(fr) 3]);
for c = 1:3
    frc(:, :, c) = fr;
end
for i = 1:length(maxheat)
    for c = 1:3
        frc(kk(maxheat(i)), :, c) = c ~= i;
    end
end
frc = imresize(frc(end:-1:1, :, :), [size(frc, 1) * z2xy, size(frc, 2)]);
frc = imresize(frc, sz(1) / max(size(frc)));
if size(frc, 1) < size(frc, 2)
    offs = floor((size(frc, 2) - size(frc, 1)) / 2);
    frontal(1 + offs:offs + size(frc, 1), :, :) = frc;
else
    offs = floor((size(frc, 1) - size(frc, 2)) / 2);
    frontal(:, 1 + offs:offs + size(frc, 2), :) = frc;
end

d = 10;

cslices = cell(1, 3);
for i = 1:3
    rgb = zeros(size(im3, 1), size(im3, 2), 3);
    for c = 1:3
        rgb(:, :, c) = selslices(end:-1:1, :, i);
        rgb(:, [1:1 + d end - d:end], c) = c ~= i;
        rgb([1:1 + d end - d:end], :, c) = c ~= i;
    end
    cslices{i} = rgb;
end

big = [frontal cslices{1}; cslices{2} cslices{3}];
% big = imresize(big, 0.5);

rgb = cell(1, 3);
for i = 1:3
    rrr = hsv2rgb(reshape(hsv(:, :, i, :), [size(im3, 1) size(im3, 2) 3]));
    for c = 1:3
        chnl = rrr(:, :, c);
        ss = selslices(:, :, i);
        chnl(isnan(chnl)) = ss(isnan(chnl));
        rrr(:, :, c) = chnl;
        
        rrr(:, [1:1 + d end - d:end], c) = c ~= i;
        rrr([1:1 + d end - d:end], :, c) = c ~= i;
    end
    rgb{i} = rrr(end:-1:1, :, :);
end

bigc = [frontal rgb{1}; rgb{2} rgb{3}];
% bigc = imresize(bigc, 0.5);