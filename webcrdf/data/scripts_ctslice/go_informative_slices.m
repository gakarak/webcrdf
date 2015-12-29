
dr0 = '';
dr1 = '';

id =  1;
sl = -1;

% % im3 = analyze75read([dr0 sprintf('id%03i.hdr', id)]);
% % info = analyze75info([dr0 sprintf('id%03i.hdr', id)]);
im3 = load_nii([dr0 sprintf('%04i.nii.gz', id)]);
z2xy = im3.hdr.dime.dim(4) / im3.hdr.dime.dim(2);
sgm = load_nii([dr1 sprintf('%04i.nii.gz_segm.nii.gz', id)]);

[slwnd, slwndc, spwise, spwisec, cbir, cbirc] = genCTpreviews(im3.img, sgm.img, z2xy);

%     subplot(321), imshow(slwnd);
%     subplot(322), imshow(slwndc);
%     subplot(323), imshow(spwise);
%     subplot(324), imshow(spwisec);
%     subplot(325), imshow(cbir);
%     subplot(326), imshow(cbirc);

dr = sprintf('infoslice/ALL/');
mkdir(dr);
imwrite(cbir, sprintf('%sid%03i_CBIR.png', dr, id));
imwrite(cbirc, sprintf('%sid%03i_CBIR_color.png', dr, id));
imwrite(slwnd, sprintf('%sid%03i_SlWnd_16_0.3_lib8.png', dr, id));
imwrite(slwndc, sprintf('%sid%03i_SlWnd_16_0.3_lib8_color.png', dr, id));
imwrite(spwise, sprintf('%sid%03i_SpWise_16_0.3_lib8.png', dr, id));
imwrite(spwisec, sprintf('%sid%03i_SpWise_16_0.3_lib8_color.png', dr, id));