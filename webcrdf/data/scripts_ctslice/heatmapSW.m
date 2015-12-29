function cmhm = heatmapSW(im, msk, mn, pc, allB, spsz, spreg, spdct)

stp = 64;
xx = 64:stp:size(im, 2) - 1 - 64;
yy = 64:stp:size(im, 1) - 1 - 64;

cmhm = zeros(size(im));
for k = 1:size(im, 3)
    ht = zeros(length(yy), length(xx));
    for j = 1:length(xx)
        for i = 1:length(yy)
            ii = yy(i);
            jj = xx(j);
            if msk(ii, jj, k)
%                 fprintf('Inspecting at (%i, %i)\n', ii, jj);
                part = im(ii - 64 + 1:ii + 64, jj - 64 + 1:jj + 64, k);

                [~, ~, desc] = superdescribe(part, spsz, spreg, [], spdct);
                if sum(desc(:)) > 0
                    desc = advNormalize(desc(:)', []);
                    desc = desc - mn;

                    x = desc * pc;
                    pr = mnrval(allB, x);
                else
                    pr = [1 0];
                end

                ht(i, j) = pr(2);

%                 ht1 = imresize(ht, [size(im, 1) size(im, 2)]);
%                 ht2 = ht1 * 0;
%                 ht2(1:end - 32, 1:end - 32) = ht1(1 + 32:end, 1 + 32:end);
%                 hsv = zeros(size(im, 1), size(im, 2), 3);
%                 ht2(ht2 < 0) = 0;
%                 ht2(ht2 > 1) = 1;
%                 hsv(:, :, 1) = 0.01 + 0.7 * (1 - ht2);
%                 hsv(:, :, 2) = msk(:, :, k);
%                 hsv(:, :, 3) = im(:, :, k) + 0.2;
%                 imshow(hsv2rgb(hsv)), drawnow;
            end
        end
    end

    ht1 = imresize(ht, [size(im, 1) size(im, 2)]);
    ht2 = ht1 * 0;
    ht2(1:end - 32, 1:end - 32) = ht1(1 + 32:end, 1 + 32:end);

    ht2(ht2 > 1) = 1;
    ht2(ht2 < 0) = 0;
    cmhm(:, :, k) = ht2;
end