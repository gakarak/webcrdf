function hsv = preparehsv(mp, im1)
% -0.5 < mp < 0.5

hsv = ones(size(im1, 1), size(im1, 2), size(im1, 3), 3);
hsv(:, :, :, 3) = im1;
h = im1 * 0;
s = im1 * 0;
h(isnan(h)) = 0.36;
s(isnan(s)) = 0;    

mp2 = mp;
h(~isnan(im1)) = 0.36 - 0.7 * mp2(~isnan(im1));

for k = 1:size(h, 3)
%     s(:, :, k) = imfilter(s(:, :, k), fspecial('gaussian', 41, 20));
        h(:, :, k) = imfilter(h(:, :, k), fspecial('gaussian', 41, 20));
end

h(h < 0) = 0.01;
h(h > 0.7) = 0.7;
s(s < 0) = 0;
s(s > 1) = 1;

hsv(:, :, :, 1) = h;
hsv(:, :, :, 2) = 1;