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

clear ImageFile LabelFile;

%% Preprocessing

S_img = size(train_images);
S_img_t = size(test_images);

y = zeros(S_img(1), 10);
y_t = zeros(S_img_t(1), 10);

for i = 1:10 
    y(:,i) = (train_labels == i-1);
    y_t(:,i) = (test_labels == i-1);
end

clear i;
 %% SVM Training
 % Linear
% svms_bin_l = cell(10,3);
% svms_bin_r = cell(10,1);
svms = cell(1,3);

% for i = 1:10 
%     svm_module_2_l = svmtrain(y(:,i), train_images, '-t 0 -c 2');
%     svm_module_4_l = svmtrain(y(:,i), train_images, '-t 0 -c 4');
%     svm_module_8_l = svmtrain(y(:,i), train_images, '-t 0 -c 8');
%     svms_bin_l{i,1} = svm_module_2_l;
%     svms_bin_l{i,2} = svm_module_4_l;
%     svms_bin_l{i,3} = svm_module_8_l;
% end
 
% % Radial
% for i = 1:10
%     svm_module_2 = svmtrain(y(:,i), train_images, '-c 2 -g 0.0625');
%     svms_bin_r{i,1} = svm_module_2;
% end

svms{1,1} = svmtrain(train_labels, train_images, '-t 0 -c 2');
svms{1,2} = svmtrain(train_labels, train_images, '-t 0 -c 4');
svms{1,3} = svmtrain(train_labels, train_images, '-t 0 -c 8');
% svm_r = svmtrain(train_labels, train_images, '-c 2 -g 0.0625');

%% SVM Testing

% svms_bin_results_l = cell(10,3);
% svms_result_l = cell(2,3);
% 
% svms_bin_results_r = zeros(10,1);

% Linear
% for i = 1:10
%     [~, pred_labels_2_l, ~] = svmpredict(y_t(:,i), test_images, svms_bin_l{i,1});
%     [~, pred_labels_4_l, ~] = svmpredict(y_t(:,i), test_images, svms_bin_l{i,2});
%     [~, pred_labels_8_l, ~] = svmpredict(y_t(:,i), test_images, svms_bin_l{i,3});
%     svms_bin_results_l{i,1} = pred_labels_2_l;
%     svms_bin_results_l{i,2} = pred_labels_4_l;
%     svms_bin_results_l{i,3} = pred_labels_8_l;
% end

% Radial
% for i = 1:10
%     [~, pred_labels_2, ~] = svmpredict(y_t(:,i), test_images, svms_bin_r{i,1});
%     svms__bin_results_r{i,1} = pred_labels_2;
% end

% [~, svms_result_l{1,1}, ~] = svmpredict(test_labels, test_images, svms{1,1});
% [~, svms_result_l{1,2}, ~] = svmpredict(test_labels, test_images, svms{1,2});
% [~, svms_result_l{1,3}, ~] = svmpredict(test_labels, test_images, svms{1,3});
[~, svms_results_r, ~] = svmpredict(test_labels, test_images, svm_r);

%% Data Gathering
% Need PoE, # of SV, 3 most important SV x+ and x-, overall error

linear_isv = zeros(10,6);

for i=1:10
    
    svm_module_2_l = svms_bin_r{i,1};
%     svm_module_4_l = svms_bin_l{i,2};
%     svm_module_8_l = svms_bin_l{i,3};
    
    sv_2_coefs = svm_module_2_l.sv_coef;
%     sv_4_coefs = svm_module_4_l.sv_coef;
%     sv_8_coefs = svm_module_8_l.sv_coef;
    
    [~, linear_isv(i,1,1)] = max(sv_2_coefs);
    sv_2_coefs(linear_isv(i,1,1)) = NaN;
    [~, linear_isv(i,2,1)] = max(sv_2_coefs);
    sv_2_coefs(linear_isv(i,2,1)) = NaN;
    [~, linear_isv(i,3,1)] = max(sv_2_coefs);
    sv_2_coefs(linear_isv(i,3,1)) = NaN;

    [~, linear_isv(i,4,1)] = min(sv_2_coefs);
    sv_2_coefs(linear_isv(i,4,1)) = NaN;
    [~, linear_isv(i,5,1)] = min(sv_2_coefs);
    sv_2_coefs(linear_isv(i,5,1)) = NaN;
    [~, linear_isv(i,6,1)] = min(sv_2_coefs);
    sv_2_coefs(linear_isv(i,6,1)) = NaN;
    
%     [~, linear_isv(i,1,2)] = max(sv_4_coefs);
%     sv_4_coefs(linear_isv(i,1,2)) = NaN;
%     [~, linear_isv(i,2,2)] = max(sv_4_coefs);
%     sv_4_coefs(linear_isv(i,2,2)) = NaN;
%     [~, linear_isv(i,3,2)] = max(sv_4_coefs);
%     sv_4_coefs(linear_isv(i,3,2)) = NaN;
% 
%     [~, linear_isv(i,4,2)] = min(sv_4_coefs);
%     sv_4_coefs(linear_isv(i,4,2)) = [];
%     [~, linear_isv(i,5,2)] = min(sv_4_coefs);
%     sv_4_coefs(linear_isv(i,5,2)) = [];
%     [~, linear_isv(i,6,2)] = min(sv_4_coefs);
%     sv_4_coefs(linear_isv(i,6,2)) = [];
%     
%     [~, linear_isv(i,1,3)] = max(sv_8_coefs);
%     sv_8_coefs(linear_isv(i,1,3)) = NaN;
%     [~, linear_isv(i,2,3)] = max(sv_8_coefs);
%     sv_8_coefs(linear_isv(i,2,3)) = NaN;
%     [~, linear_isv(i,3,3)] = max(sv_8_coefs);
%     sv_8_coefs(linear_isv(i,3,3)) = NaN;
% 
%     [~, linear_isv(i,4,3)] = min(sv_8_coefs);
%     sv_8_coefs(linear_isv(i,4,3)) = NaN;
%     [~, linear_isv(i,5,3)] = min(sv_8_coefs);
%     sv_8_coefs(linear_isv(i,5,3)) = NaN;
%     [~, linear_isv(i,6,3)] = min(sv_8_coefs);
%     sv_8_coefs(linear_isv(i,6,3)) = NaN;
    
end

for i = 1:10
    
    svm_module_2_l = svms_bin_r{i,1};
%     svm_module_4_l = svms_bin_l{i,2};
%     svm_module_8_l = svms_bin_l{i,3};
%     
    sv_2_indices = svm_module_2_l.sv_indices;
%     sv_4_indices = svm_module_4_l.sv_indices;
%     sv_8_indices = svm_module_8_l.sv_indices;
    
    linear_isv(i,1,1) = sv_2_indices(linear_isv(i,1,1));
    linear_isv(i,2,1) = sv_2_indices(linear_isv(i,2,1));
    linear_isv(i,3,1) = sv_2_indices(linear_isv(i,3,1));
    linear_isv(i,4,1) = sv_2_indices(linear_isv(i,4,1));
    linear_isv(i,5,1) = sv_2_indices(linear_isv(i,5,1));
    linear_isv(i,6,1) = sv_2_indices(linear_isv(i,6,1));
    
%     linear_isv(i,1,2) = sv_4_indices(linear_isv(i,1,2));
%     linear_isv(i,2,2) = sv_4_indices(linear_isv(i,2,2));
%     linear_isv(i,3,2) = sv_4_indices(linear_isv(i,3,2));
%     linear_isv(i,4,2) = sv_4_indices(linear_isv(i,4,2));
%     linear_isv(i,5,2) = sv_4_indices(linear_isv(i,5,2));
%     linear_isv(i,6,2) = sv_4_indices(linear_isv(i,6,2));
%     
%     linear_isv(i,1,3) = sv_8_indices(linear_isv(i,1,3));
%     linear_isv(i,2,3) = sv_8_indices(linear_isv(i,2,3));
%     linear_isv(i,3,3) = sv_8_indices(linear_isv(i,3,3));
%     linear_isv(i,4,3) = sv_8_indices(linear_isv(i,4,3));
%     linear_isv(i,5,3) = sv_8_indices(linear_isv(i,5,3));
%     linear_isv(i,6,3) = sv_8_indices(linear_isv(i,6,3));
    
end

sample_x = zeros(10,S_img(2),6,1);
for i = 1:10
    
    sample_x(i,:,1,1) = train_images(linear_isv(i,1,1),:);
    sample_x(i,:,2,1) = train_images(linear_isv(i,2,1),:);
    sample_x(i,:,3,1) = train_images(linear_isv(i,3,1),:);
    sample_x(i,:,4,1) = train_images(linear_isv(i,4,1),:);
    sample_x(i,:,5,1) = train_images(linear_isv(i,5,1),:);
    sample_x(i,:,6,1) = train_images(linear_isv(i,6,1),:);
    
%     sample_x(i,:,1,2) = train_images(linear_isv(i,1,2),:);
%     sample_x(i,:,2,2) = train_images(linear_isv(i,2,2),:);
%     sample_x(i,:,3,2) = train_images(linear_isv(i,3,2),:);
%     sample_x(i,:,4,2) = train_images(linear_isv(i,4,2),:);
%     sample_x(i,:,5,2) = train_images(linear_isv(i,5,2),:);
%     sample_x(i,:,6,2) = train_images(linear_isv(i,6,2),:);
%     
%     sample_x(i,:,1,3) = train_images(linear_isv(i,1,3),:);
%     sample_x(i,:,2,3) = train_images(linear_isv(i,2,3),:);
%     sample_x(i,:,3,3) = train_images(linear_isv(i,3,3),:);
%     sample_x(i,:,4,3) = train_images(linear_isv(i,4,3),:);
%     sample_x(i,:,5,3) = train_images(linear_isv(i,5,3),:);
%     sample_x(i,:,6,3) = train_images(linear_isv(i,6,3),:);
    
end

for i = 1:10
    for k = 1:1
        digit = [];
        for j = 1:6
            digs = reshape(sample_x(i,:,j,k), [28,28]);
            digit = [digit,digs'];
        end
        figure;
        imagesc(digit);
        colormap gray;
    end
end

clear i j k digs;
%% Margins

margins = cell(10,3);
for i = 1:10
    
    for j = 1:3
        
        svm = svms_bin_l{i,j};
        coefs = svm.sv_coef;
        ind = svm.sv_indices;
        rho = svm.rho;
        coef = zeros(S_img(1),1);
        coef(ind) = coefs;
        samples = train_images(ind,:);
        
        w = coef.*y(:,i);
        w = w'*train_images;
        
        b = 0;
        for k = 1:length(ind)
            b = b + w*samples(k,:)';
        end
        
        b = rho - b;
        b = b.*ones(S_img(1),1);
        margin = y(:,i)'.*(w*train_images' + b');
        margins{i,j} = margin;        
    end
end

%% CDF

c_dists = cell(10, 1);
c_edges = cell(10,1);

for i = 1:10
    
    dists = cell(5, 1);
    edges = cell(5, 1);
    
    for j = 1:3
        
        margin = margins{i,j};
        margin(margin == 0) = [];
        p = histogram(margin);
        
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
    dists = c_dists{i};
    edges = c_edges{i};
    
    figure;
    hold on;
    title(['CDF for Digit: ', num2str(i-1)]);
    dist = dists{1};
    edge = edges{1};
    subplot(3,1,1);
    plot(edge, dist);
    ylabel('Probability');
    dist = dists{2};
    edge = edges{2};
    subplot(3,1,2);
    plot(edge, dist);
    ylabel('Probability');
    dist = dists{3};
    edge = edges{3};
    subplot(3,1,3);
    plot(edge, dist); 
    ylabel('Probability');
    axis('tight');
    
    
end
    

        
            
        
        
        
        