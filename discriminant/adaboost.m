function [g, g_t, max_w, vis] = adaboost(g, g_t, y, x, x_t, t, vis)
    
    w = exp(-y .* g);
    
    [~, max_w] = max(w);
    
    alphas = gradient(t, w, x, y);
    
    [w_t, t_min] = min(alphas(:, 1));
    [w_t2, t_min2] = min(alphas(:,3));
    
    twin = 0;
    
    if w_t2 < w_t
        w_t = w_t2;
        t_min = t_min2;
        twin = 1;
    end
    
    w_t = step_size(w_t, w);
    
    if twin == 0
        j = alphas(t_min, 2);
        vis(j) = 255;
    else
        j = alphas(t_min, 4);
        vis(j) = 0;
    end
    
    x_j = (x(:, j) >= t(t_min));
    x_j_t = (x_t(:, j) >= t(t_min));
    
    x_j = x_j - ~x_j;
    x_j_t = x_j_t - ~x_j_t;
    
    if twin == 1
        x_j = -x_j;
        x_j_t = -x_j_t;
    end
    
    g = g + w_t .* x_j;
    g_t = g_t + w_t .* x_j_t;
    
end
    
    
    
    
    
    
    
    
    
    
    