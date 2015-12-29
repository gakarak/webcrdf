function maxheat = maxheatslices(heatmap, kid)

ss = heatmap;
ss(ss < 0) = 0;
ss(isnan(ss)) = 0;
ht = sum(sum(ss, 1), 2);
ht = ht(:);
maxheat = [0 0 0];
for i = 1:3
    ht_ = ht;
    ht_(kid ~= i) = 0;
    maxheat(i) = find(ht_ == max(ht_), 1);
end