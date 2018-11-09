function alphas = gradient(t, w, x, y)

    S_x = size(x);

    alphas = zeros(length(t), 4);
    
    for i = 1:length(t)
        
        temp_log = (x >= t(i));
        temp_log = temp_log - ~temp_log;
        twin_temp_log = -temp_log;
        
        for j = 1:S_x(2)
            
            temp_log(:, j) = (temp_log(:, j) ~= y);
            twin_temp_log(:, j) = (twin_temp_log(:, j) ~= y);
            
        end
                
        temp_log = bsxfun(@times, w, temp_log);
        twin_temp_log = bsxfun(@times, w, twin_temp_log);
        
        temp = sum(temp_log, 1);
        twin_temp = sum(twin_temp_log, 1);
        
        [alphas(i,1), alphas(i, 2)] = min(temp);
        [alphas(i,3), alphas(i, 4)] = min(twin_temp);
        
    end
    
end