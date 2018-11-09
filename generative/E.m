function weights = E(data, alphas, mus, sigmas)

    S = size(data);
    weights = zeros(length(alphas), S(1));
    
    hk = zeros(S(1), 1);
    
    for i = 1:length(alphas)

    hk = hk + alphas(i) .* mvnpdf(data, mus(i,:), sigmas(i,:));
            
    end
    
    for i = 1:length(alphas)
        
        hj = mvnpdf(data, mus(i,:), sigmas(i,:));
        
        hj = alphas(i) .* hj;
        
        hj = hj ./ hk;
    
        weights(i, :) = hj';
        
    end
        
end