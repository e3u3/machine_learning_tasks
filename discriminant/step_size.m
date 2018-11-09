function w_t = step_size(w_top, w)

    w_bot = sum(w);
    
    epsilon = w_top/w_bot;
    
    w_t = (1/2) * log((1-epsilon)/epsilon);
    
end
