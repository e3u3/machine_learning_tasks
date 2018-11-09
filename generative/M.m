function [alphas, mus, sigmas] = M(data, weights)

    S = size(weights);
    L = size(data);
    
    alphas = zeros(S(1), 1);
    mus = zeros(S(1), L(2));
    sigmas = zeros(S(1), L(2));
    new_data = zeros(L(1), L(2));
    
    for i = 1:S(1)
        
        weight_new = sum(weights(i, :));
        
        alphas(i) = weight_new/S(2);
        
        for j = 1:L(1)
            
            new_data(j, :) = weights(i, j) .* data(j, :);
        
        end
        
        sum_data = sum(new_data);
        
        mus(i, :) = sum_data ./ weight_new;
        
        variance = zeros(L(2), L(2));
        
        for j = 1:L(1)
            
            variance = variance + weights(i, j) * (data(j, :) - mus(i, :))' * (data(j, :) - mus(i, :));
            
        end
        
        variance = variance ./ weight_new;
        
        sigmas(i, :) = diag(variance);
        
        for j = 1:L(2)
            
            if(sigmas(i, j) < 0.001)
                sigmas(i, j) = 0.001;
            end
            
        end

    end
    

end