function ht = heatmapRCBIR(cls, im)

ht = im * 0;
for k = 1:size(im, 3)
    iiid = cooccur2D(im(:, :, k), [0 1], 12, 1:5, 3);
    desc = advNormalize(iiid, 5);
    dm = pdist2(desc, cls, 'cityblock');
    dm = dm(dm > 0);
    sims = sort(dm);
    mn = mean(sims(1:20));
    
    ht(:, :, k) = mn;
end
ht(isnan(im)) = 0;