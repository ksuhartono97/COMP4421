function img_result = dft_2d(img_input, flag)

if strcmp(flag, 'DFT')
    [m, n] = size(img_input);
    wm = zeros(m,m);
    wn = zeros(n,n);
    
    img = zeros([m,n]);
    
    for y = 1 : m
        for x = 1 : n
            img(y,x) = double(img_input(y,x)) * power(-1, x+y);
        end
    end           
    
    for u = 0 : (m-1) 
        for x = 0 : (m-1)
            wm(u+1,x+1) = exp(-2 * pi * 1i * ((u * x) / m));
        end
    end
    
    for v = 0 : (n-1) 
        for y = 0 : (n-1)
            wn(y+1,v+1) = exp(-2 * pi * 1i * ((v * y) / n));
        end
    end
    
    img_result = wm * double(img) * wn;
    
elseif strcmp(flag, 'IDFT')
%     img_result = zeros(size(img_input));
    [m, n] = size(img_input);
    img_input = img_input * power(-1, m+n);
    wm = zeros(m,m);
    wn = zeros(n,n);
    
    for u = 0 : (m-1) 
        for x = 0 : (m-1)
            wm(u+1,x+1) = exp(2 * pi * 1i * ((u * x) / m));
        end
    end
    
    for v = 0 : (n-1) 
        for y = 0 : (n-1)
            wn(y+1,v+1) = exp(2 * pi * 1i * ((v * y) / n));
        end
    end
    
    img_result = wm * double(img_input) * wn;
end

% You should include the center shifting in this function