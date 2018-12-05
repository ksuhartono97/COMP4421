function img_result = medfilt2d(img_input, sz)
[ysize, xsize] = size(img_input);

img = im2double(img_input);
r = floor(sz/2);
resized_mat(1:ysize + r*2, 1:xsize+r*2) = 0;
resized_mat(1+r:ysize+r,1+r:xsize+r) = img;
for i = 1: r
    resized_mat(i, 1+r:xsize+r) = img(1, 1:xsize);
    resized_mat(1+r:ysize+r, i) = img(1:ysize, 1);
    resized_mat(ysize+i, 1+r:xsize+r) = img(ysize, 1:xsize);
    resized_mat(1+r:ysize+r, xsize+i) = img(1:ysize, xsize);
end

[y_res, x_res] = size(resized_mat);

img_result(1:ysize, 1:xsize) = 0;

for i=1+r : y_res - r
    for j=1+r : x_res - r
        window_mat = resized_mat((i - r):(i + r),(j - r):(j + r ));
        vect = window_mat(:);
        med = median(vect) * 255;
        img_result(i-r,j-r) = med;
    end
end

% for i=1+r : x_res - r
%     for j=1+r : y_res - r
%         window_mat = resized_mat((j - r):(j + r),(i - r):(i + r ));
%         vect = window_mat(:);
%         
%         img_result(j-r,i-r) = median(vect);
%     end
% end

img_result = uint8(img_result);