function img_result = filter_spa(img_input, filter)
[ysize, xsize] = size(img_input);
[yfilt, xfilt] = size(filter);

img = im2double(img_input);
r = floor(yfilt/2);
resized_mat(1:ysize + r*2, 1:xsize+r*2) = 0;
resized_mat(1+r:ysize+r,1+r:xsize+r) = img;

[y_res, x_res] = size(resized_mat);

img_result(1:ysize, 1:xsize) = 0;


% implementation of bonus

for i = 0 : (xfilt * yfilt) - 1
    kernel_col = 1 + int32(mod(i, xfilt));
    kernel_row = 1 + int32(floor(i/xfilt));
    kernel_val = filter(kernel_row, kernel_col);
    
    im_mat = resized_mat(kernel_row : kernel_row + y_res - yfilt, kernel_col :kernel_col +  x_res-xfilt);
    
    img_result = img_result + kernel_val * im_mat;
    
end

img_result = img_result * 255;
img_result = uint8(img_result);



% implementation of regular code
% 
% for i=1+r : y_res - r
%     for j=1+r : x_res - r
%         im_mat = resized_mat((i - r):(i + r),(j - r):(j + r ));
%         res = im_mat.*filter;
%         res = res*255;
%         img_result(i-r,j-r) = sum(res(:));
%     end
% end
% img_result = uint8(img_result);