function [scores, mn, pc, allB] = spscores(descs, cls)
%Calculates correspondence score [-1; 1] for each superpixel class

descs = advNormalize(descs, []);
mn = mean(descs);
[pc, sc0, ev] = pca(descs);
nprin = 1 + sum(cumsum(ev) < 0.95 * sum(ev));
nprin = min(10, nprin);
sc = sc0(:, 1:nprin);
[~, ps] = corr(cls, sc);
jj = find(ps < 0.01);
X = sc(:, jj);

B = mnrfit(X, cls + 1);
% prs = mnrval(B, X);
% prs = prs(:, 2);
    
% [x, y, ~, auc] = perfcurve(cls, prs, 1);
% plot(x, y, 'b', [0 1], [0 1], 'g:');
% title(sprintf('AUC = %.3f by PCs: %s', auc, num2str(jj)));

% rs = corr(descs, cls);
allB = zeros(size(sc0, 2), 1);
allB(jj) = B(2:end);
scores = - pc * allB / size(sc0, 2);

allB = [B(1); allB];
% p = mnrval(allB, (descs - repmat(mn, size(descs, 1), 1)) * pc);
% p = p(:, 2);
% [x, y, ~, auc] = perfcurve(cls, p, 1);
% plot(x, y, 'b', [0 1], [0 1], 'g:');
% title(sprintf('AUC = %.3f by PCs: %s', auc, num2str(jj)));