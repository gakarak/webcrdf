function [kk, kid] = candidateslices(segm)

bn = segm;
prfl = sum(sum(bn, 1), 2);
prfl = prfl(:);

sls = zeros(1, int16(sum(prfl) / 1e3));
for k = 1:length(prfl)
    sls(1 + sum(sls > 0):sum(sls > 0) + round(prfl(k) / 1e3)) = k;
end

q = round(quantile(sls, [0.3 0.6 0.9]));
% prfl = imfilter(prfl, ones(1, 11) / 11);

kk = [q(1) - 4, q(1) - 2, q(1), q(1) + 2, q(1) + 4, ...
    q(2) - 4, q(2) - 2, q(2), q(2) + 2, q(2) + 4, ...
    q(3) - 4, q(3) - 2, q(3), q(3) + 2, q(3) + 4];
kid = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3];
kid = kid(kk > 0 & kk <= size(segm, 3));
kk = kk(kk > 0 & kk <= size(segm, 3));