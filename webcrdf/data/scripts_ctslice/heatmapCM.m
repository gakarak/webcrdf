function cmm0 = heatmapCM(scores, ftMaps)
N = sqrt(length(scores));
cmsc = reshape(scores, [N N]);

spmap0 = ftMaps{7} + 1;
clmap0 = ftMaps{8};
cmm0 = clmap0 * 0;
for k = 1:size(spmap0, 3)
    msk = ~isnan(spmap0(:, :, k));
    jj = max(1, find(sum(msk) > 3, 1, 'first') - 16)...
            :min(size(spmap0, 2), find(sum(msk) > 0, 1, 'last') + 16);
    ii = max(1, find(sum(msk, 2) > 3, 1, 'first') - 16)...
        :min(size(spmap0, 1), find(sum(msk, 2) > 0, 1, 'last') + 16);
    
    disp(num2str(k));
    spmap = spmap0(ii, jj, k);
    clmap = clmap0(ii, jj, k);

    if sum(clmap(:) > 0) > 0
        cmm = clmap * 0;
        cls = zeros(1, max(spmap(:)) - min(spmap(:)) + 1);
        for i = 1:max(spmap(:))
            s = spmap == i;
            cls(i) = mean(clmap(s(:)));
        end

        for i = 1:max(spmap(:))
            s = spmap == i;
            cl1 = cls(i);

            if cl1 > 0
                s = bwmorph(s, 'dilate', 1) & ~s;
                s = s & (spmap > i);
                
                sp2s = unique(spmap(s(:)));
                sp2s = sp2s(cls(sp2s) > 0);
                
                for j = 1:length(sp2s)
                    toadd = cmsc(cl1, cls(sp2s(j)));
%                     toadd = sign(toadd) * toadd^2;
                    cmm(spmap == i) = cmm(spmap == i) + toadd;
                    cmm(spmap == sp2s(j)) = cmm(spmap == sp2s(j)) + toadd;
        %             subplot(121); imshow((spmap == i) * 2 + (spmap == sp2s(j)), []);
        %             subplot(122); imshow(cmm, []);
        %             drawnow;
                end
            end
        end

        cmm0(ii, jj, k) = cmm;
    end
end