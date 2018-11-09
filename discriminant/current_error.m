function [error, t_error] = current_error(g, g_t, y, y_t)
    
    h = sign(g);
    h_t = sign(g_t);
    
    difference = (h ~= y);
    difference_t = (h_t ~= y_t);
    
    error = sum(difference)/size(difference, 1);
    
    t_error = sum(difference_t)/size(difference_t, 1);
    
end
    