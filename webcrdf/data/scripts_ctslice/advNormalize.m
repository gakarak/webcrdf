function b = advNormalize(a, dims)
% performes advanced descriptor matrix normalization so that for each
% distance (D) dimension sum of elements is equal to 1 (normalization is
% performed separately for each distance D). Here 'a' is initial descriptor
% matrix (one matrix == one row), and dims is vector of dimensions + the
% last element representing the number of distance (D) dimension.

if nargin < 2 || isempty(dims);
    b = scal(a', zeros(1, size(a', 2)), sum(a, 2)')';
    return
end

if numel(dims) == 1
    b = a;
    n = size(b, 2);
    
    if mod(n, dims) ~= 0
        throw(MException('INPUT:BadParameter', ...
            'numel(a) must be devisible by dims'));
    end
    
    for i = 1:dims
        jj = 1 + (i - 1) * n / dims:i * n / dims;
        b(:, jj) = scal(b(:, jj)', zeros(1, ...
            size(b(:, jj)', 2)), sum(b(:, jj), 2)')';
    end
    return
end

ds = zeros(dims(1:end-1));
for i=1:numel(ds)
    [s1 s2 s3 s4 s5 s6 s7 s8 s9 s10] = ind2sub(dims(1:end-1), i);
    sb = [s1 s2 s3 s4 s5 s6 s7 s8 s9 s10];
    ds(i) = sb(dims(end));
end
ds = permute(ds, numel(dims)-1:-1:1);
ds = ds(:);

b = a * 0;
for i=1:dims(dims(end))
    b(:, ds == i) = scal(a(:, ds == i)', zeros(1, ...
        size(a(:, ds == i)', 2)), sum(a(:, ds == i), 2)')';
end

% useful for debug. explores cross-distance correlation of descriptor
% elements
% rw = b(1, :);
% figure, scatter(rw(ds == 1), rw(ds == 2))

end