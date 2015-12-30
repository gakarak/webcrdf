function go_informative_slices2(pathImgInp, pathImgOut)
    im3 = load_nii(pathImgInp);
    z2xy = im3.hdr.dime.dim(2) / im3.hdr.dime.dim(4);
    sgm = load_nii(sprintf('%s_segm.nii.gz', pathImgInp(1:end-7)));
    image = permute(im3.img + 1024, [2 1 3]);
    segm = permute(sgm.img, [2 1 3]);
    [slwnd, slwndc, spwise, spwisec, cbir, cbirc] = genCTpreviews(im3.img, sgm.img, z2xy);
    imwrite(slwndc, pathImgOut);
end
% %     imwrite(cbir, sprintf('%sid%03i_CBIR.png', dr, id));
% %     imwrite(cbirc, sprintf('%sid%03i_CBIR_color.png', dr, id));
% %     imwrite(slwnd, sprintf('%sid%03i_SlWnd_16_0.3_lib8.png', dr, id));
% %     imwrite(slwndc, sprintf('%sid%03i_SlWnd_16_0.3_lib8_color.png', dr, id));
% %     imwrite(spwise, sprintf('%sid%03i_SpWise_16_0.3_lib8.png', dr, id));
% %     imwrite(spwisec, sprintf('%sid%03i_SpWise_16_0.3_lib8_color.png', dr, id));
