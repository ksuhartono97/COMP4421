function f = gaussian(size, cutoff)
if length(size) == 1
    rows = size; 
    cols = size;
else
    rows = size(1); 
    cols = size(2);
end


% if(mod(M,2) == 0)
%     cM = floor(M/2) + 0.5;
% else
%     cM = floor(M/2) + 1;
% end
% if(mod(N,2) == 0)
%     cN = floor(N/2) + 0.5;
% else
%     cN = floor(N/2) + 1;
% end
% 
% a = [1:M];
% b = [1:N];
% A = repmat(a',1,N);
% B = repmat(b,M,1);
% A = (A-cM).^2;
% B = (B-cN).^2;

if mod(cols,2)
    xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
else
    xrange = [-cols/2:(cols/2-1)]/cols;
end

if mod(rows,2)
    yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
else
    yrange = [-rows/2:(rows/2-1)]/rows;
end
[x,y] = meshgrid(xrange, yrange);
radius = sqrt(x.^2 + y.^2);        
  

f = 1.0 - (exp(-(radius.^2)./(2*cutoff^2)));