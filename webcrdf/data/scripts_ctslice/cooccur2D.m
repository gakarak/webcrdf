function desc = cooccur2D(m, iRange, iBins, dists, nDots)

m = double(m);

b = (m - iRange(1)) / (iRange(2) - iRange(1));
b = int8(floor(b * iBins) + 1);
b(b < 1) = 1;
b(b > iBins) = iBins;
b(isnan(m)) = -1;

LUT1 = calcLUT(dists);

hh = cell(1, size(LUT1, 1));
desc = zeros(1, length(dists) * iBins^nDots);
for i = 1:size(LUT1, 1);
% for id = 1:length(dists)
    LUT = LUT1;
    id = find(dists == LUT(i, 1), 1);
    b0 = b;
    
    rr = LUT(:, 1) == dists(id);
%     lut = LUT(LUT(:, 1) == dists(id), :);
    
    r = [max([0; -LUT(rr, 3); -LUT(rr, 5)]), max([0; LUT(rr, 3); ...
        LUT(rr, 5)]), max([0; -LUT(rr, 2); -LUT(rr, 4)]), ...
        max([0; LUT(rr, 2); LUT(rr, 4)])];
    
    b0 = b0(1 + r(1):end - r(2), 1 + r(3):end - r(4));

    bs = cell(1, nDots - 1);
    for ndot = 2:nDots
        d = (ndot - 2) * 2;
        b1 = b(1 + r(1) + LUT(i, 3 + d):end - r(2) + LUT(i, 3 + d), ...
            1 + r(3) + LUT(i, 2 + d):end - r(4) + LUT(i, 2 + d));   
        bs{ndot - 1} = b1;
    end

    cobins = int8(zeros(size(b0, 1), size(b0, 2), nDots));
    cobins(:, :, 1) = b0;
    for ndot = 2:nDots
        cobins(:, :, ndot) = bs{ndot - 1};
    end

    cobins = sort(cobins, 3);
    mn = min(cobins, [], 3);

    cumul = ones(size(cobins, 1), size(cobins, 2));
    for ndot = 1:nDots
        cumul = cumul + (double(cobins(:, :, ndot)) - 1) * iBins^(ndot - 1);
    end
    cumul(mn < 0) = -1;

    h = hist(cumul(cumul(:) > 0), 1:(iBins^nDots));
    h2 = zeros(1, length(desc));
    h2(1 + (id - 1) * iBins^nDots:id * iBins^nDots) = h;
    hh{i} = h2;
%     desc(1 + (id - 1) * iBins^nDots:id * iBins^nDots) = ...
%         desc(1 + (id - 1) * iBins^nDots:id * iBins^nDots) + h;
end

for i = 1:size(LUT1, 1)
    desc = desc + hh{i};
end

function LUT = calcLUT(dists)

LUT = zeros(max(dists)^2 * 16, 5);
s3 = sqrt(3);

y = 0;
il = 0;
for x = 1:max(dists)
    d = round(sqrt(x^2 + y^2));
    if ~isempty(find(dists == d, 1))
        id = find(dists == d, 1);
        il = il + 1;
        x1 = round(x / 2 - s3 / 2 * y);
        y1 = round(y / 2 + s3 / 2 * x);
        LUT(il, :) = [dists(id), x, y, x1, y1];
    end
end

for x = -max(dists):max(dists)
    for y = 1:max(dists)
        d = round(sqrt(x^2 + y^2));
        if ~isempty(find(dists == d, 1))
            id = find(dists == d, 1);
            il = il + 1;
            x1 = round(x / 2 - s3 / 2 * y);
            y1 = round(y / 2 + s3 / 2 * x);
            LUT(il, :) = [dists(id), x, y, x1, y1];
        end
    end
end

LUT = LUT(LUT(:, 1) > 0, :);
[~, idx] = sort(LUT(:, 1));
LUT = LUT(idx, :);