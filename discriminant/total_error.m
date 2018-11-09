function [error, t_error] = total_error(g, g_t, y, y_t, c, c_t)
    
    error = zeros(1, size(g, 2));

    t_error = zeros(1, size(g, 2));
    
    for i = 1:length(y)
        
        [~, class] = max(g(i, :));
        
        class = class - 1;
        
        if i < length(y_t)
            
            [~, class_t] = max(g_t(i, :));
        
            class_t = class_t - 1;
            
            if class_t ~= y_t(i)
                
                t_error(y_t(i) + 1) = t_error(y_t(i) + 1) + 1;                
                
            end
            
        end
        
        if class ~= y(i)
            
            error(y(i) + 1) = error(y(i) + 1) + 1;
            
        end
        
    end
    
    error = error./c;
    
    t_error = t_error./c_t;
    
end