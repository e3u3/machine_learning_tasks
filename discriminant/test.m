%%%%%%%%%%%%%%%%%%%%%%
% ECE271B - ML II
% HW 3
% Ibrahim Akbar
% Winter 2018
% UCSD
%%%%%%%%%%%%%%%%%%%%%%

%% Loading Data

clear;

ImageFile = gunzip('.\training_set\train-images-idx3-ubyte.gz');
LabelFile = gunzip('.\training_set\train-labels-idx1-ubyte.gz');
[train_images, train_labels] = readMNIST(ImageFile{1}, LabelFile{1}, 20000, 0);

ImageFile = gunzip('.\test_set\t10k-images-idx3-ubyte.gz');
LabelFile = gunzip('.\test_set\t10k-labels-idx1-ubyte.gz');
[test_images, test_labels] = readMNIST(ImageFile{1}, LabelFile{1}, 10000, 0);
S_img = size(train_images);
S_img_t = size(test_images);

clear ImageFile LabelFile;

%% AdaBoost

% Loss Function: exponential: exp(-g(x))

T = 250;
classes = 1;
N = 50;

t = (1:N);
t = t./N;

w = zeros(S_img(1), classes);
g = zeros(S_img(1), classes);
g_t = zeros(S_img_t(1), classes);
alpha = zeros(N, 2);

z = [5, 10, 50, 100, 250];
loc = 1;

margins = zeros(S_img(1), classes, length(z));
max_w = zeros(T, classes);

bin_test_errors = zeros(T, classes);
tot_test_errors = zeros(T, 1);

logical = zeros(S_img(1), S_img(2), N);

bin_train_errors = zeros(T, classes);
tot_train_errors = zeros(T, 1);

y = zeros(S_img(1), classes);
y_t = zeros(S_img_t(1), classes);

c = zeros(classes, 1);
c_t = zeros(classes, 1);

for j = 1:classes

    y(:, j) = (train_labels + 1 == j);
    c(j) = sum(y(:, j));
    y(:, j) = y(:, j) + ~y(:, j) * (-1);

    y_t(:, j) = (test_labels + 1 == j);
    c_t(j) = sum(y_t(:, j));
    y_t(:, j) = y_t(:, j) + ~y_t(:, j) * (-1);
    
end
    
for k = 1:N
    
   logical(:, :, k) = (train_images >= t(k));
   logical(:, :, k) = logical(:, :, k) + ~logical(:, :, k) * (-1);
   
end

for i = 1:T
   
    for k = 1:classes
        
        for j = 1:S_img(1)
        
            w(j, k) = exp(-y(j, k) * g(j, k));
            
            for l = 1:N
                
                logical(j, :, l) = y(j, k) * logical(j, :, l) * w(j, k)
                
                
            
            if j < S_img_t(1)

            end
            
        end
        
    end
end