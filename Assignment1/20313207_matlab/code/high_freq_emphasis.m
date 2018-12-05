function img_result = high_freq_emphasis(img_input, a, b, type)

% [asd, def] = size(img_input);
if strcmp(type, 'butterworth')
    f = butterworth(size(img_input), 0.1, 1);
%     figure,imshow(f)
elseif strcmp(type, 'gaussian')
    f = gaussian(size(img_input), 0.1);
%     figure,imshow(f)
end

f = a + b .* f;

dft_img = dft_2d(img_input, 'DFT');

filtered_dft = dft_img .* f ;

idft_res = dft_2d(filtered_dft, 'IDFT');
idft_res = abs(idft_res);
img_result = mat2gray(idft_res);
