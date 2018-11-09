function r = range(x)
   
    max = 0;
    min = 999999999;
    for i = 2:length(x)
        if(x(i) > max)
            max = x(i);
        end
        if(x(i) < min)
            min = x(i);
        end
    end
    r = max-min;