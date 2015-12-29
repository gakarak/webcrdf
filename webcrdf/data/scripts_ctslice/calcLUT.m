function params = calcLUT(params0)
% fills of params.comatrixParams.LUT data field

params = params0;
dists = params.comatrixParams.dists;
LUT = zeros(max(dists)^2 * 16, 3);

j = 0;
il = 0;
for i=1:max(dists)
    d = round(sqrt(i^2 + j^2));
    if ~isempty(find(dists == d, 1))
        id = find(dists == d, 1);
        il = il + 1;
        LUT(il, :) = [dists(id), i, j];
    end
end

for i=-max(dists):max(dists)
    for j=1:max(dists)
        d = round(sqrt(i^2 + j^2));
        if ~isempty(find(dists == d, 1))
            id = find(dists == d, 1);
            il = il + 1;
            LUT(il, :) = [dists(id), i, j];
        end
    end
end

params.comatrixParams.LUT = LUT;

end