%%%%%%%%%%%%%%%%%%%%%%
% ECE271B - ML II
% HW 3
% Ibrahim Akbar
% Winter 2018
% UCSD
%%%%%%%%%%%%%%%%%%%%%%

%% Loading Data

clear;

ImageFile = gunzip('./training_set/train-images-idx3-ubyte.gz');
LabelFile = gunzip('./training_set/train-labels-idx1-ubyte.gz');
[train_images, train_labels] = readMNIST(ImageFile{1}, LabelFile{1}, 20000, 0);

ImageFile = gunzip('./test_set/t10k-images-idx3-ubyte.gz');
LabelFile = gunzip('./test_set/t10k-labels-idx1-ubyte.gz');
[test_images, test_labels] = readMNIST(ImageFile{1}, LabelFile{1}, 10000, 0);
S_img = size(train_images);
S_img_t = size(test_images);

clear ImageFile LabelFile;

%% AdaBoost

T = 250;
classes = 10;
N = 50;

z = [5, 10, 50, 100, 250];
loc = 1;

t = (1:N);
t = t./N;

g = zeros(S_img(1), classes);
g_t = zeros(S_img_t(1), classes);

y = zeros(S_img(1), classes);
c = zeros(1, classes);
y_t = zeros(S_img_t(1), classes);
c_t = zeros(1, classes);

errors = cell(T, 2);

margins = zeros(S_img(1), classes, 5);
max_w = zeros(T, classes);
vis = ones(classes, S_img(2));
vis = vis * 128;


for i = 1:classes
    
    y(:, i) = (train_labels == i - 1);
    c(i) = sum(y(:, i));
    y(:, i) = y(:, i) - ~y(:, i);
    
    y_t(:, i) = (test_labels == i - 1);
    c_t(i) = sum(y_t(:,i));
    y_t(:, i) = y_t(:, i) - ~y_t(:, i);
    
end

for i = 1:T
    
    for j = 1:classes
        
        [g(:, j), g_t(:, j), max_w(i, j), vis] = adaboost(g(:, j), g_t(:, j), y(:, j), train_images, test_images, t, vis);
        
        if i == z(loc)
            margins(:, j, loc) = y(:, j) .* g(:, j);
        end
        
    end
    
    if i == z(loc)
        loc = loc + 1;
    end
    
    [error_t, t_error_t] = total_error(g, g_t, train_labels, test_labels, c, c_t);
    
    errors{i, 1} = error_t;
    errors{i, 2} = t_error_t;
    
    disp(['ERROR: ', num2str(error_t), ' TEST ERROR: ', num2str(t_error_t), ' TIME: ', num2str(i)]);
end
        
%% CDF

c_dists = cell(10, 1);
c_edges = cell(10,1);

for i = 1:classes
    
    dists = cell(5, 1);
    edges = cell(5, 1);
    
    for j = 1:5
        
        p = histogram(margins(:, i, j));
        
        edge = p.BinEdges;
        
        dist = zeros(length(edge), 1);
        
        
        for k = 1:length(edge)-1
            
            x = (p.Data >= edge(k)) & (p.Data <= edge(k+1));
            
            if k == 1
                dist(k+1) = sum(x)/length(p.Data);
                continue;
                
            else
                dist(k+1) = sum(x)/length(p.Data) + dist(k);
            end
            
        end
        
        dists{j} = dist;
        edges{j} = edge;
    end
    
    c_dists{i} = dists;
    
    c_edges{i} = edges;
    
end
    

%%
for i = 1:10
    
    figure(i);
    hold on;
    
    dists = c_dists{i};
    edges = c_edges{i};
    
    for j = 1:5
        
        dist = dists{j};
        edge = edges{j};
        
        plot(edge, dist);
        
    end
    
    axis('tight');
    
    title(['CDF for Digit: ', num2str(i-1)]);
    xlabel('Margin');
    ylabel('Probability');
    legend('5', '10', '50', '100', '250', 'Location', 'Best');
    
end
    
    
%%

for i = 1:10
    
    figure(i);
    
    plot(1:T, max_w(:,i));
    
    xlabel('Iterations');
    ylabel('Sample');
    title('Max Weighted Sample Per Iteration');
    
end
                
