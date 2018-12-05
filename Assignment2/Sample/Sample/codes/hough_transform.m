function [img_marked, corners] = hough_transform(img)

% Implement the Hough transform to detect the target A4 paper
% Input parameter:
% .    img - original input image
% .    (You can add other input parameters if you need. If you have added
% .    other input parameters, please state for what reasons in the PDF file)
% Output parameter:
% .    img_marked - image with marked sides and corners detected by Hough transform
% .    corners - the 4 corners of the target A4 paper

fil = 1/49*ones(7);

gray_img = rgb2gray(img);
gray_img = im2double(gray_img);
img_ave = filter2(fil, gray_img);
img_med = medfilt2(img_ave);

img_dog = imgaussfilt(img_med,1) - imgaussfilt(img_med, 5);
img_dog = imadjust(img_dog);

img_recrop = img_dog(15+1:end-15, 15+1:end-15);  
img_dog = padarray(img_recrop,[15 15],0);

BW = imbinarize(img_dog);

SE = strel('disk', 5);
BW2 = imdilate(BW, SE);

BW2 = bwareaopen(BW2, 400);
BW2 = imdilate(BW2, SE);
BW2 = imfill(BW2,'holes');
BW2 = imerode(BW2, SE);

BW2 = bwmorph(BW2,'remove');
SE = strel('disk', 3);
BW2 = imdilate(BW2, SE);

BW2 = bwareafilt(BW2, 1);
% BW2 = bwskel(BW2);
img_processed = bwskel(BW2);

% help
[h, w] = size(img_processed);
sh = norm([h,w]) + 1;
thetas = 0 : pi/180 : pi;
[y, x] = find(img_processed);

s = sin(thetas);
c = cos(thetas);
rho = [x,y] * [c; s];
rho = floor(rho + sh);

H = full(sparse(rho, repmat(1 : length(thetas), [length(x), 1]), 1));

[~, H_indexes] = sort(H(:), 'descend');
[r, t] = ind2sub(size(H), H_indexes(1:72));

H1 = H;

filtered_r = [];
filtered_t = [];

for i = 1:length(t)
    hasParallel = false;
    hasParallelCount = 0;
    for j = 1:length(t)
        if i == j
            continue
        end
        if abs(t(i) - t(j)) <= 10
            hasParallelCount = hasParallelCount + 1;
        end
        if hasParallelCount == 6
            hasParallel = true;
            break;
        end
    end
    if hasParallel == true
        filtered_r(end+1) = r(i);
        filtered_t(end+1) = t(i);
    end
end

filtered_r2 = [filtered_r(1)];
filtered_t2 = [filtered_t(1)];

filter_size = size(H1)/5;
r_filter_thresh = filter_size(1);

for i = 2:length(filtered_r)
    pushflag = true;
    for j = 1 : length(filtered_r2)
        t_diff = abs(filtered_t2(j) - filtered_t(i));
        
        if t_diff > 160
            t_diff = abs(t_diff - 180);
        end
        
        r_diff = abs(filtered_r2(j) - filtered_r(i));
        
        if(t_diff < 7 && r_diff < r_filter_thresh)
            pushflag = false;
        end
        
        if pushflag == false
            break;
        end       
       
    end
    if pushflag == true
        filtered_r2(end + 1) = filtered_r(i);
        filtered_t2(end + 1) = filtered_t(i);
    end
    
    if length(filtered_r2) == 4
        break
    end
end

r = filtered_r2;
t = filtered_t2;

%UNCOMMENT HERE
r = (r - sh);
hough_lines = zeros(2,2,4);

line_pairs = zeros(1, 2, 4);

% figure, imshow(img), hold on
for i = 1:length(r)
  rho = r(i);
  theta = t(i);
  theta_calc = theta/length(thetas) * (max(thetas) - min(thetas));
  
  line_pairs(:, :, i) = [rho, theta_calc];
  
  xy_okay = [0 0 0 0];
  
  y1 = rho/sin(theta_calc);
  if y1 > 0 && y1 <= h
    xy_okay(1) = 1;
  end
  
  y2 = (rho - w*cos(theta_calc))/sin(theta_calc);
  if y2 > 0 && y2 <= h
    xy_okay(2) = 1;
  end
  
  x3 = rho / cos(theta_calc);
  if x3 > 0 && x3 <= w
    xy_okay(3) = 1;
  end
  
  x4 = (rho - h*sin(theta_calc)) / cos(theta_calc);
  if x4 > 0 && x4 <= w
    xy_okay(4) = 1;
  end
  
  xy = [1 y1; w y2; x3 1; x4 h];
  
  points = xy(find(xy_okay), :);
  hough_lines(:,:,i) = [points(:, 1), points(:, 2)];
%   line(points(:, 1), points(:, 2), 'color', 'g', 'linewidth', 1);
end
% hold on

% line_pairs
first_line = line_pairs(:,:, 1);
similar_pair_index = 1;
theta_diff = 360;
for i = 2:4
    fl_theta = first_line(:, 2)*180/pi;
    sl_theta = line_pairs(:, 2, i)*180/pi;
    t_diff = abs(fl_theta - sl_theta);
    if t_diff > 160
        t_diff = abs(t_diff - 180);
    end
    
    if theta_diff > t_diff
        theta_diff = t_diff;
        similar_pair_index = i;
    end    
end

rlp = zeros(1, 2, 4);
rlp(:, :, 1) = line_pairs(:, :, 1);
rlp(:, :, 2) = line_pairs(:, :, similar_pair_index);
resorted_line_pairs_ni = 3;
for i = 2:4
    if i == similar_pair_index
        continue
    end
    rlp(:, :, resorted_line_pairs_ni) = line_pairs(:, :, i);
    resorted_line_pairs_ni = resorted_line_pairs_ni +1;
end

corners = [];

for i = 1:2
    for j = 3:4
        eq_theta = [cos(rlp(:,2,i)) sin(rlp(:,2,i)); cos(rlp(:,2,j)) sin(rlp(:,2,j))];
        eq_rho = [rlp(:,1,i);rlp(:,1,j)];
%         inv(eq_theta) * eq_rho
%         corners(:, :, corners_ind) = inv(eq_theta)*eq_rho;
%         corners_ind = corners_ind + 1;
        inters = inv(eq_theta)*eq_rho;
        point = [inters(1) inters(2)];
        corners = [corners; point];
    end
end

% hold on
% plot(corners(:, 1), corners(:,2), 'go', 'MarkerSize', 5, 'MarkerFaceColor', 'g');
% hold off

mark_mask = zeros([size(BW2)]);

for i = 1:4
    x_coord = corners(i, 1);
    y_coord = corners(i, 2);
    for j = 1:359
        x_r = 10 * cos(i/pi);
        y_r = 10 * sin(i/pi);
        xfc = floor(x_coord + x_r);
        yfc = floor(y_coord + y_r);
        mark_mask(yfc, xfc) = 255;
    end
end

SE = strel('disk', 20);
mark_mask = imdilate(mark_mask, SE);

for i = 1:4
  hough_lines(:,:,i);  
  points_a = hough_lines(1,:,i);
  points_b = hough_lines(2,:,i);
%   F(t) = (1-t)[x1;y1] + t[x2;y2;];
  t_step = 1000;
  for t = 1:t_step
      ratio = t/t_step;
      first_part = (1 - ratio) * [points_a(1); points_a(2)];
      second_part = ratio * [points_b(1); points_b(2)];
      res = first_part + second_part;
      yfc = floor(res(2));
      xfc = floor(res(1));
      mark_mask(yfc, xfc) = 255;
  end
end

SE = strel('disk', 10);
mark_mask = imdilate(mark_mask, SE);
mark_mask = uint8(mark_mask);

img_marked = img;
img_marked(:,:,2) = img_marked(:,:,2) + mark_mask;
% figure, imshow(img_marked)

% img_marked = BW2;
