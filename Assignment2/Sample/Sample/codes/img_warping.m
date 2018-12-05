function img_warp = img_warping(img, corners, n)

% Implement the image warping to transform the target A4 paper into the
% standard A4-size paper
% Input parameter:
% .    img - original input image
% .    corners - the 4 corners of the target A4 paper detected by the Hough transform
% .    (You can add other input parameters if you need. If you have added
% .    other input parameters, please state for what reasons in the PDF file)
% Output parameter:
% .    img_warp - the standard A4-size target paper obtained by image warping
% .    n - determine the size of the result image

scorn = zeros(4, 2);
min_cd = pdist([0,0;corners(1,:)], 'euclidean');
max_cd = pdist([0,0;corners(1,:)], 'euclidean');
% min_corner = corners(1,:)
% max_corner = corners(1,:)
min_ci = 1;
max_ci = 1;

for i = 2:length(corners)
    this_dist = pdist([0,0;corners(i,:)], 'euclidean');
    if(this_dist < min_cd)
        min_cd = this_dist;
        min_ci = i;
    end
    if(this_dist > max_cd)
        max_cd = this_dist;
        max_ci = i;
    end
end

scorn(1, :) = corners(min_ci, :);
scorn(4, :) = corners(max_ci, :);

for i=1:length(corners)
    if(i == min_ci || i == max_ci)
        continue
    end
    if(corners(i,1) > corners(i, 2))
        scorn(2, :) = corners(i, :);
    end
    if(corners(i,1) < corners(i, 2))
        scorn(3, :) = corners(i, :);
    end
end

ori_height = sqrt((scorn(3, 1) - scorn(1, 1)).^2 + (scorn(1, 2) - scorn(3, 2)).^2);
ori_width = sqrt((scorn(2, 1) - scorn(1, 1)).^2 + (scorn(1, 2) - scorn(2, 2)).^2);

targ_h = 0;
targ_w = 0;

%vertical
if ori_height > ori_width
    targ_h = 297*n;
    targ_w = 210*n;
else
    %horizontal
    targ_h = 210*n;
    targ_w = 297*n;
end

% T1 = [0 0];
% T2 = [0 width];
% T3 = [height 0];
% T4 = [height width];
% T_corns = [0 0; 0 targ_w; targ_h 0; targ_h targ_w];

x_or = scorn(:, 1);
y_or = scorn(:, 2);
xy_or = x_or .* y_or;
id_or = [1;1;1;1];

A = [x_or, y_or, xy_or, id_or];
invA = inv(A);

x_targ = [1; targ_w; 1; targ_w];
y_targ = [1; 1; targ_h; targ_h];

% x_param = invA * x_targ;
% y_param = invA * y_targ;

xy_targ = x_targ .* y_targ;

rev_A = [x_targ, y_targ, xy_targ, id_or];
inv_revA = inv(rev_A);

tar_x_param = inv_revA * x_or;
tar_y_param = inv_revA * y_or;

% testx = x_param(1) * height + x_param(2) * height + x_param(3) * height * height + x_param(4)
% testy = y_param(1) * width + y_param(2) * width + y_param(3) * width*width + y_param(4)

reverse_map = zeros(floor(targ_h), floor(targ_w), 2);

for i = 1:floor(targ_h)
    for j = 1:floor(targ_w)
        u = j;
        v = i;
        uv = u*v;
        x = tar_x_param(1) * u + tar_x_param(2) * v + tar_x_param(3) * uv + tar_x_param(4);
        y = tar_y_param(1) * u + tar_y_param(2) * v + tar_y_param(3) * uv + tar_y_param(4);
        point = [x, y];
        reverse_map(i,j,:) = point;
    end
end

img_warp = zeros([targ_h targ_w 3]);
% size(img_warp)

for i = 1:targ_h
    for j = 1:targ_w
        x = reverse_map(i, j, 1);
        y = reverse_map(i, j, 2);
        
        minx = floor(x);
        maxx = ceil(x);
        miny = floor(y);
        maxy = ceil(y);
%         aratio = (x - minx) * (y-miny);
        
        val = ((x - minx) * (y-miny)) * img(maxy, maxx, :);
        val = val + ((maxx - x) * (y-miny))*img(maxy, minx, :);
        val = val + ((x - minx) *(maxy - y))*img(miny, minx, :);
        val = val + ((maxx - x) * (maxy - y))*img(miny, minx, :);
        
        img_warp(i, j, :) = val;
    end
end


img_warp = uint8(img_warp);


% img_warp = img;