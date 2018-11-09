%%%%%%%%%%%%%%%%%%%%%%
% ECE271C - ML III
% HW 1
% Ibrahim Akbar
% Spring 2018
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

%% Sampling Data

p = 10;
samples = 1000;
sample = randi([1,20000],1,samples);
data = train_images(sample,:);
data = data';
one = ones(samples,1);

%% PCA Matrix For Loop

mu_non = zeros(S_img(2),1);

non_start = tic;

for i = 1:samples
    mu_non = mu_non + data(:,i);
end
mu_non = (1/samples).*mu_non;

center_images = zeros(S_img(2),samples);

for i = 1:samples
   center_images(:,i) = data(:,i) - mu_non; 
end

cov_non = zeros(S_img(2),S_img(2));

for i = 1:samples
    cov_non = cov_non + center_images(:,i)*center_images(:,i)';
end

cov_non = (1/samples).*cov_non;

[V,D] = eig(cov_non);

[~, inds] = sort(diag(D),'descend');

Ds = D(inds,inds);
Vs = V(:,inds);
pc = Vs(:,1:p);

time_non = toc(non_start);

clear non_start;
%% PCA Semi-Vectorized

semi_start = tic;

mu_semi = (1/samples)*data*one;

center_images = zeros(S_img(2),samples);

for i = 1:samples
   center_images(:,i) = data(:,i) - mu_semi; 
end

cov_semi = (1/samples)*(center_images*center_images');


[V,D] = eig(cov_semi);

[~, inds] = sort(diag(D),'descend');

Ds = D(inds,inds);
Vs = V(:,inds);
pc = Vs(:,1:p);
time_semi =toc(semi_start);

clear semi_start;

%% PCA Vectorized

full_start = tic;

mu_full = (1/samples)*data*one;

center_images = data - mu_full*one';

cov_full = (1/samples)*(center_images*center_images');

[V,D] = eig(cov_full);

[~, inds] = sort(diag(D),'descend');

Ds = D(inds,inds);
Vs = V(:,inds);
pc = Vs(:,1:p);
time_full = toc(full_start);

clear full_start;

%% PCA Vectorized Differently

full_start2 = tic;

cov_full2 = (1/samples)*data*(eye(samples,samples) - (1/samples)*(one*one'))*data';

[V,D] = eig(cov_full2);

[~, inds] = sort(diag(D),'descend');

Ds = D(inds,inds);
Vs = V(:,inds);
pc = Vs(:,1:p);
time_full2 = toc(full_start2);

clear full_start2;

%% Eigenvalue, Eigenvector, Ratio Plots

if(cov_non == cov_semi)
    disp('YES');
end
ratio = zeros(S_img(2),1);
eigen = diag(Ds);

for i = 1:S_img(2)
    ratio(i) = sum(eigen(1:i))/sum(eigen);
end

figure(1);
stem(eigen);
% set(gca, 'yscal', 'log');
xlabel('Index: i');
ylabel('Eigenvalue');
title('Sorted Eigenvalues');

figure(2);
stem(ratio);
% set(gca, 'yscal', 'log');
xlabel('K');
ylabel('Ratio');
title('Ratio: r_{k}');

figure(3);
subplot(4,4,1);
imagesc(reshape(Vs(:,1), [28,28]));
subplot(4,4,2);
imagesc(reshape(Vs(:,2), [28,28]));
subplot(4,4,3);
imagesc(reshape(Vs(:,3), [28,28]));
subplot(4,4,4);
imagesc(reshape(Vs(:,4), [28,28]));
subplot(4,4,5);
imagesc(reshape(Vs(:,5), [28,28]));
subplot(4,4,6);
imagesc(reshape(Vs(:,6), [28,28]));
subplot(4,4,7);
imagesc(reshape(Vs(:,7), [28,28]));
subplot(4,4,8);
imagesc(reshape(Vs(:,8), [28,28]));
subplot(4,4,9);
imagesc(reshape(Vs(:,9), [28,28]));
subplot(4,4,10);
imagesc(reshape(Vs(:,10), [28,28]));
subplot(4,4,11);
imagesc(reshape(Vs(:,11), [28,28]));
subplot(4,4,12);
imagesc(reshape(Vs(:,12), [28,28]));
subplot(4,4,13);
imagesc(reshape(Vs(:,13), [28,28]));
subplot(4,4,14);
imagesc(reshape(Vs(:,14), [28,28]));
subplot(4,4,15);
imagesc(reshape(Vs(:,15), [28,28]));
subplot(4,4,16);
imagesc(reshape(Vs(:,16), [28,28]));
colormap(gray);

%% PCA Reconstructed Image

samples = 16;
sample = randi([1,10000],1,samples);
test_data = test_images(sample,:);
test_data = test_data';
one = ones(samples,1);

test_mu = (1/samples)*test_data*one;

c_test_data = test_data -test_mu*one';

r_1 = find(ratio > .4);
r_2 = find(ratio > .7);
r_3 = find(ratio > .9);

p_1 = r_1(1);
p_2 = r_2(1);
p_3 = r_3(1);

pc_1 = Vs(:,1:p_1);
pc_2 = Vs(:,1:p_2);
pc_3 = Vs(:,1:p_3);

pc_test_data_1 = pc_1'*c_test_data;
pc_test_data_2 = pc_2'*c_test_data;
pc_test_data_3 = pc_3'*c_test_data;

test_data_1_recon = pc_1*pc_test_data_1 + test_mu;
test_data_2_recon = pc_2*pc_test_data_2 + test_mu;
test_data_3_recon = pc_3*pc_test_data_3 + test_mu;

figure(4);
subplot(4,4,1);
imagesc(reshape(test_data(:,1), [28,28])');
subplot(4,4,2);
imagesc(reshape(test_data(:,2), [28,28])');
subplot(4,4,3);
imagesc(reshape(test_data(:,3), [28,28])');
subplot(4,4,4);
imagesc(reshape(test_data(:,4), [28,28])');
subplot(4,4,5);
imagesc(reshape(test_data(:,5), [28,28])');
subplot(4,4,6);
imagesc(reshape(test_data(:,6), [28,28])');
subplot(4,4,7);
imagesc(reshape(test_data(:,7), [28,28])');
subplot(4,4,8);
imagesc(reshape(test_data(:,8), [28,28])');
subplot(4,4,9);
imagesc(reshape(test_data(:,9), [28,28])');
subplot(4,4,10);
imagesc(reshape(test_data(:,10), [28,28])');
subplot(4,4,11);
imagesc(reshape(test_data(:,11), [28,28])');
subplot(4,4,12);
imagesc(reshape(test_data(:,12), [28,28])');
subplot(4,4,13);
imagesc(reshape(test_data(:,13), [28,28])');
subplot(4,4,14);
imagesc(reshape(test_data(:,14), [28,28])');
subplot(4,4,15);
imagesc(reshape(test_data(:,15), [28,28])');
subplot(4,4,16);
imagesc(reshape(test_data(:,16), [28,28])');
colormap(gray);

figure(5);
subplot(4,4,1);
imagesc(reshape(test_data_1_recon(:,1), [28,28])');
subplot(4,4,2);
imagesc(reshape(test_data_1_recon(:,2), [28,28])');
subplot(4,4,3);
imagesc(reshape(test_data_1_recon(:,3), [28,28])');
subplot(4,4,4);
imagesc(reshape(test_data_1_recon(:,4), [28,28])');
subplot(4,4,5);
imagesc(reshape(test_data_1_recon(:,5), [28,28])');
subplot(4,4,6);
imagesc(reshape(test_data_1_recon(:,6), [28,28])');
subplot(4,4,7);
imagesc(reshape(test_data_1_recon(:,7), [28,28])');
subplot(4,4,8);
imagesc(reshape(test_data_1_recon(:,8), [28,28])');
subplot(4,4,9);
imagesc(reshape(test_data_1_recon(:,9), [28,28])');
subplot(4,4,10);
imagesc(reshape(test_data_1_recon(:,10), [28,28])');
subplot(4,4,11);
imagesc(reshape(test_data_1_recon(:,11), [28,28])');
subplot(4,4,12);
imagesc(reshape(test_data_1_recon(:,12), [28,28])');
subplot(4,4,13);
imagesc(reshape(test_data_1_recon(:,13), [28,28])');
subplot(4,4,14);
imagesc(reshape(test_data_1_recon(:,14), [28,28])');
subplot(4,4,15);
imagesc(reshape(test_data_1_recon(:,15), [28,28])');
subplot(4,4,16);
imagesc(reshape(test_data_1_recon(:,16), [28,28])');
colormap(gray);

figure(6);
subplot(4,4,1);
imagesc(reshape(test_data_2_recon(:,1), [28,28])');
subplot(4,4,2);
imagesc(reshape(test_data_2_recon(:,2), [28,28])');
subplot(4,4,3);
imagesc(reshape(test_data_2_recon(:,3), [28,28])');
subplot(4,4,4);
imagesc(reshape(test_data_2_recon(:,4), [28,28])');
subplot(4,4,5);
imagesc(reshape(test_data_2_recon(:,5), [28,28])');
subplot(4,4,6);
imagesc(reshape(test_data_2_recon(:,6), [28,28])');
subplot(4,4,7);
imagesc(reshape(test_data_2_recon(:,7), [28,28])');
subplot(4,4,8);
imagesc(reshape(test_data_2_recon(:,8), [28,28])');
subplot(4,4,9);
imagesc(reshape(test_data_2_recon(:,9), [28,28])');
subplot(4,4,10);
imagesc(reshape(test_data_2_recon(:,10), [28,28])');
subplot(4,4,11);
imagesc(reshape(test_data_2_recon(:,11), [28,28])');
subplot(4,4,12);
imagesc(reshape(test_data_2_recon(:,12), [28,28])');
subplot(4,4,13);
imagesc(reshape(test_data_2_recon(:,13), [28,28])');
subplot(4,4,14);
imagesc(reshape(test_data_2_recon(:,14), [28,28])');
subplot(4,4,15);
imagesc(reshape(test_data_2_recon(:,15), [28,28])');
subplot(4,4,16);
imagesc(reshape(test_data_2_recon(:,16), [28,28])');
colormap(gray);

figure(7);
subplot(4,4,1);
imagesc(reshape(test_data_3_recon(:,1), [28,28])');
subplot(4,4,2);
imagesc(reshape(test_data_3_recon(:,2), [28,28])');
subplot(4,4,3);
imagesc(reshape(test_data_3_recon(:,3), [28,28])');
subplot(4,4,4);
imagesc(reshape(test_data_3_recon(:,4), [28,28])');
subplot(4,4,5);
imagesc(reshape(test_data_3_recon(:,5), [28,28])');
subplot(4,4,6);
imagesc(reshape(test_data_3_recon(:,6), [28,28])');
subplot(4,4,7);
imagesc(reshape(test_data_3_recon(:,7), [28,28])');
subplot(4,4,8);
imagesc(reshape(test_data_3_recon(:,8), [28,28])');
subplot(4,4,9);
imagesc(reshape(test_data_3_recon(:,9), [28,28])');
subplot(4,4,10);
imagesc(reshape(test_data_3_recon(:,10), [28,28])');
subplot(4,4,11);
imagesc(reshape(test_data_3_recon(:,11), [28,28])');
subplot(4,4,12);
imagesc(reshape(test_data_3_recon(:,12), [28,28])');
subplot(4,4,13);
imagesc(reshape(test_data_3_recon(:,13), [28,28])');
subplot(4,4,14);
imagesc(reshape(test_data_3_recon(:,14), [28,28])');
subplot(4,4,15);
imagesc(reshape(test_data_3_recon(:,15), [28,28])');
subplot(4,4,16);
imagesc(reshape(test_data_3_recon(:,16), [28,28])');
colormap(gray);