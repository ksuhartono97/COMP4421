function img_result = filter2d_fre(img_input, filter)

[n, m] = size(img_input);

P = 2*m;
Q = 2*n;

padded_img = zeros([Q, P]);
padded_img(1:n,1:m) = img_input;

dft_img = dft_2d(padded_img, 'DFT');
[filt_y, filt_x] = size(filter);

padded_filter = zeros([Q, P]);
padded_filter(1:filt_y, 1:filt_x) = filter;

dft_filter = dft_2d(padded_filter, 'DFT');

filtered_dft = dft_img .* dft_filter ;
idft_res = dft_2d(filtered_dft, 'IDFT');
idft_res = abs(idft_res);
idft_res = mat2gray(idft_res);

img_result = idft_res(1:n, 1:m);

% img_result = zeros(size(img_input));

