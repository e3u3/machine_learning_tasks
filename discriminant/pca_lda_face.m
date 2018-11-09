%%%%%%%%%%%%%%%%%%%%%%%%%
% ECE271B - HW 1
% Ibrahim Akbar
% Winter 2018
% UCSD
%%%%%%%%%%%%%%%%%%%%%%%%%

%% Problem 5 Set Up Training
clear;
clc;

file0 = dir('./trainset/subset0/*.jpg');
file1 = dir('./trainset/subset1/*.jpg');
file2 = dir('./trainset/subset2/*.jpg');
file3 = dir('./trainset/subset3/*.jpg');
file4 = dir('./trainset/subset4/*.jpg');
file5 = dir('./trainset/subset5/*.jpg');

img0 = [];

for file = file0'
    img = reshape(imread(['./trainset/subset0/', file.name]), [2500, 1]);
    img0 = [img0, img];
end

for file = file1'
    img = reshape(imread(['./trainset/subset1/', file.name]), [2500, 1]);
    img0 = [img0, img];
end

for file = file2'
    img = reshape(imread(['./trainset/subset2/', file.name]), [2500, 1]);
    img0 = [img0, img];
end

for file = file3'
    img = reshape(imread(['./trainset/subset3/', file.name]), [2500, 1]);
    img0 = [img0, img];
end

for file = file4'
    img = reshape(imread(['./trainset/subset4/', file.name]), [2500, 1]);
    img0 = [img0, img];
end

for file = file5'
    img = reshape(imread(['./trainset/subset5/', file.name]), [2500, 1]);
    img0 = [img0, img];
end

img0 = double(img0);

clear file0 file1 file2 file3 file4 file5 file img;

%% Test Set Up

file0 = dir('./testset/subset6/*.jpg');
file1 = dir('./testset/subset7/*.jpg');
file2 = dir('./testset/subset8/*.jpg');
file3 = dir('./testset/subset9/*.jpg');
file4 = dir('./testset/subset10/*.jpg');
file5 = dir('./testset/subset11/*.jpg');

img6 = [];
img7 = [];
img8 = [];
img9 = [];
img10 = [];
img11 = [];

for file = file0'
    img = reshape(imread(['./testset/subset6/', file.name]), [2500, 1]);
    img6 = [img6, img];
end

for file = file1'
    img = reshape(imread(['./testset/subset7/', file.name]), [2500, 1]);
    img7 = [img7, img];
end

for file = file2'
    img = reshape(imread(['./testset/subset8/', file.name]), [2500, 1]);
    img8 = [img8, img];
end

for file = file3'
    img = reshape(imread(['./testset/subset9/', file.name]), [2500, 1]);
    img9 = [img9, img];
end

for file = file4'
    img = reshape(imread(['./testset/subset10/', file.name]), [2500, 1]);
    img10 = [img10, img];
end

for file = file5'
    img = reshape(imread(['./testset/subset11/', file.name]), [2500, 1]);
    img11 = [img11, img];
end

img6 = double(img6);
img7 = double(img7);
img8 = double(img8);
img9 = double(img9);
img10 = double(img10);
img11 = double(img11);

clear file0 file1 file2 file3 file4 file5 file img;

%% PCA Training

S = size(img0);

I = eye(S(2));

X_C = (I - (1/S(2))*ones(S(2), S(2)))* img0';

[U, E, V_1_a] = svd(X_C);

pc_16 = V_1_a(:, 1:30);

figure(1);
subplot(4,4,1);
imagesc(reshape(pc_16(:, 1), [50,50]));
subplot(4,4,2);
imagesc(reshape(pc_16(:, 2), [50,50]));
subplot(4,4,3);
imagesc(reshape(pc_16(:, 3), [50,50]));
subplot(4,4,4);
imagesc(reshape(pc_16(:, 4), [50,50]));
subplot(4,4,5);
imagesc(reshape(pc_16(:, 5), [50,50]));
subplot(4,4,6);
imagesc(reshape(pc_16(:, 6), [50,50]));
subplot(4,4,7);
imagesc(reshape(pc_16(:, 7), [50,50]));
subplot(4,4,8);
imagesc(reshape(pc_16(:, 8), [50,50]));
subplot(4,4,9);
imagesc(reshape(pc_16(:, 9), [50,50]));
subplot(4,4,10);
imagesc(reshape(pc_16(:, 10), [50,50]));
subplot(4,4,11);
imagesc(reshape(pc_16(:, 11), [50,50]));
subplot(4,4,12);
imagesc(reshape(pc_16(:, 12), [50,50]));
subplot(4,4,13);
imagesc(reshape(pc_16(:, 13), [50,50]));
subplot(4,4,14);
imagesc(reshape(pc_16(:, 14), [50,50]));
subplot(4,4,15);
imagesc(reshape(pc_16(:, 15), [50,50]));
subplot(4,4,16);
imagesc(reshape(pc_16(:, 16), [50,50]));
colormap(gray);

clear S I;
%% LDA

% 0 - 1:40
% 1 - 41:80
% 2 - 81:120
% 3 - 121:160
% 4 - 161:200
% 5 - 201:240

mu0 = (1/40)*sum(img0(:,1:40),2);
mu1 = (1/40)*sum(img0(:,41:80),2);
mu2 = (1/40)*sum(img0(:,81:120),2);
mu3 = (1/40)*sum(img0(:,121:160),2);
mu4 = (1/40)*sum(img0(:,161:200),2);
mu5 = (1/40)*sum(img0(:,201:240),2);

cov0 = zeros(2500,2500);
cov1 = zeros(2500,2500);
cov2 = zeros(2500,2500);
cov3 = zeros(2500,2500);
cov4 = zeros(2500,2500);
cov5 = zeros(2500,2500);
I = eye(2500, 2500);

for i = 1:40
    cov0 = cov0 + (img0(:,i)-mu0)*(img0(:,i)-mu0)';
end
cov0 = cov0/40;

for i = 41:80
    cov1 = cov1 + (img0(:,i)-mu1)*(img0(:,i)-mu1)';
end
cov1 = cov1/40;

for i = 81:120
    cov2 = cov2 + (img0(:,i)-mu2)*(img0(:,i)-mu2)';
end
cov2 = cov2/40;

for i = 121:160
    cov3 = cov3 + (img0(:,i)-mu3)*(img0(:,i)-mu3)';
end
cov3 = cov3/40;

for i = 161:200
    cov4 = cov4 + (img0(:,i)-mu4)*(img0(:,i)-mu4)';
end
cov4 = cov4/40;

for i = 201:240
    cov5 = cov5 + (img0(:,i)-mu5)*(img0(:,i)-mu5)';
end
cov5 = cov5/40;

w01 = (cov0 + cov1 + I)\(mu0 - mu1);
w02 = (cov0 + cov2 + I)\(mu0 - mu2);
w03 = (cov0 + cov3 + I)\(mu0 - mu3);
w04 = (cov0 + cov4 + I)\(mu0 - mu4);
w05 = (cov0 + cov5 + I)\(mu0 - mu5);
w12 = (cov1 + cov2 + I)\(mu1 - mu2);
w13 = (cov1 + cov3 + I)\(mu1 - mu3);
w14 = (cov1 + cov4 + I)\(mu1 - mu4);
w15 = (cov1 + cov5 + I)\(mu1 - mu5);
w23 = (cov2 + cov3 + I)\(mu2 - mu3);
w24 = (cov2 + cov4 + I)\(mu2 - mu4);
w25 = (cov2 + cov5 + I)\(mu2 - mu5);
w34 = (cov3 + cov4 + I)\(mu3 - mu4);
w35 = (cov3 + cov5 + I)\(mu3 - mu5);
w45 = (cov4 + cov5 + I)\(mu4 - mu5);

figure(2);
subplot(4,4,1);
imagesc(reshape(w01, [50,50]));
subplot(4,4,2);
imagesc(reshape(w02, [50,50]));
subplot(4,4,3);
imagesc(reshape(w03, [50,50]));
subplot(4,4,4);
imagesc(reshape(w04, [50,50]));
subplot(4,4,5);
imagesc(reshape(w05, [50,50]));
subplot(4,4,6);
imagesc(reshape(w12, [50,50]));
subplot(4,4,7);
imagesc(reshape(w13, [50,50]));
subplot(4,4,8);
imagesc(reshape(w14, [50,50]));
subplot(4,4,9);
imagesc(reshape(w15, [50,50]));
subplot(4,4,10);
imagesc(reshape(w23, [50,50]));
subplot(4,4,11);
imagesc(reshape(w24, [50,50]));
subplot(4,4,12);
imagesc(reshape(w25, [50,50]));
subplot(4,4,13);
imagesc(reshape(w34, [50,50]));
subplot(4,4,14);
imagesc(reshape(w35, [50,50]));
subplot(4,4,15);
imagesc(reshape(w45, [50,50]));
colormap(gray);

w_t = zeros(2500, 15);
w_t(:,1) = w01;
w_t(:,2) = w02;
w_t(:,3) = w03;
w_t(:,4) = w04;
w_t(:,5) = w05;
w_t(:,6) = w12;
w_t(:,7) = w13;
w_t(:,8) = w14;
w_t(:,9) = w15;
w_t(:,10) = w23;
w_t(:,11) = w24;
w_t(:,12) = w25;
w_t(:,13) = w34;
w_t(:,14) = w35;
w_t(:,15) = w45;

clear mu0 mu1 mu2 mu3 mu4 mu5 cov0 cov1 cov2 cov3 cov4 cov5 I i ...
    w01 w02 w03 w04 w05 w12 w13 w14 w15 w23 w24 w25 w34 w35 w45;

%% Training PCA Projections Learning

z0 = pc_16(:,1:15)'*img0(:,1:40);
z1 = pc_16(:,1:15)'*img0(:,41:80);
z2 = pc_16(:,1:15)'*img0(:,81:120);
z3 = pc_16(:,1:15)'*img0(:,121:160);
z4 = pc_16(:,1:15)'*img0(:,161:200);
z5 = pc_16(:,1:15)'*img0(:,201:240);

z_mu0 = (1/40)*sum(z0,2);
z_mu1 = (1/40)*sum(z1,2);
z_mu2 = (1/40)*sum(z2,2);
z_mu3 = (1/40)*sum(z3,2);
z_mu4 = (1/40)*sum(z4,2);
z_mu5 = (1/40)*sum(z5,2);

z_cov0 = zeros(15, 15);
z_cov1 = zeros(15, 15);
z_cov2 = zeros(15, 15);
z_cov3 = zeros(15, 15);
z_cov4 = zeros(15, 15);
z_cov5 = zeros(15, 15);

for i = 1:40
    z_cov0 = z_cov0 + (z0(:,i) - z_mu0)*(z0(:,i) - z_mu0)';
end
z_cov0 = z_cov0/40;

for i = 1:40
    z_cov1 = z_cov1 + (z1(:,i) - z_mu1)*(z1(:,i) - z_mu1)';
end
z_cov1 = z_cov1/40;

for i = 1:40
    z_cov2 = z_cov2 + (z2(:,i) - z_mu2)*(z2(:,i) - z_mu2)';
end
z_cov2 = z_cov2/40;

for i = 1:40
    z_cov3 = z_cov3 + (z3(:,i) - z_mu3)*(z3(:,i) - z_mu3)';
end
z_cov3 = z_cov3/40;

for i = 1:40
    z_cov4 = z_cov4 + (z4(:,i) - z_mu4)*(z4(:,i) - z_mu4)';
end
z_cov4 = z_cov4/40;

for i = 1:40
    z_cov5 = z_cov5 + (z5(:,i) - z_mu5)*(z5(:,i) - z_mu5)';
end
z_cov5 = z_cov5/40;

mu_t = {z_mu0,z_mu1,z_mu2,z_mu3,z_mu4,z_mu5};
cov_t = {z_cov0,z_cov1,z_cov2,z_cov3,z_cov4,z_cov5};

clear z0 z1 z2 z3 z4 z5 z_mu0 z_mu1 z_mu2 z_mu3 z_mu4 z_mu5 ...
    z_cov0 z_cov1 z_cov2 z_cov3 z_cov4 z_cov5 i;
%% PCA Test Projection

z6 = pc_16(:,1:15)'*img6;
z7 = pc_16(:,1:15)'*img7;
z8 = pc_16(:,1:15)'*img8;
z9 = pc_16(:,1:15)'*img9;
z10 = pc_16(:,1:15)'*img10;
z11 = pc_16(:,1:15)'*img11;

%% PCA Classification

ec1 = 0;
for i = 1:10
     i0 = (z6(:,i) - mu_t{1})'*inv(cov_t{1})*(z6(:,i)-mu_t{1})+log(det(cov_t{1}));
     i1 = (z6(:,i) - mu_t{2})'*inv(cov_t{2})*(z6(:,i)-mu_t{2})+log(det(cov_t{2}));
     i2 = (z6(:,i) - mu_t{3})'*inv(cov_t{3})*(z6(:,i)-mu_t{3})+log(det(cov_t{3}));
     i3 = (z6(:,i) - mu_t{4})'*inv(cov_t{4})*(z6(:,i)-mu_t{4})+log(det(cov_t{4}));
     i4 = (z6(:,i) - mu_t{5})'*inv(cov_t{5})*(z6(:,i)-mu_t{5})+log(det(cov_t{5}));
     i5 = (z6(:,i) - mu_t{6})'*inv(cov_t{6})*(z6(:,i)-mu_t{6})+log(det(cov_t{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 1
         ec1 = ec1 + 1;
     end
end

ec2 = 0;
for i = 1:10
     i0 = (z7(:,i) - mu_t{1})'*inv(cov_t{1})*(z7(:,i)-mu_t{1})+log(det(cov_t{1}));
     i1 = (z7(:,i) - mu_t{2})'*inv(cov_t{2})*(z7(:,i)-mu_t{2})+log(det(cov_t{2}));
     i2 = (z7(:,i) - mu_t{3})'*inv(cov_t{3})*(z7(:,i)-mu_t{3})+log(det(cov_t{3}));
     i3 = (z7(:,i) - mu_t{4})'*inv(cov_t{4})*(z7(:,i)-mu_t{4})+log(det(cov_t{4}));
     i4 = (z7(:,i) - mu_t{5})'*inv(cov_t{5})*(z7(:,i)-mu_t{5})+log(det(cov_t{5}));
     i5 = (z7(:,i) - mu_t{6})'*inv(cov_t{6})*(z7(:,i)-mu_t{6})+log(det(cov_t{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 2
         ec2 = ec2 + 1;
     end
end

ec3 = 0;
for i = 1:10
     i0 = (z8(:,i) - mu_t{1})'*inv(cov_t{1})*(z8(:,i)-mu_t{1})+log(det(cov_t{1}));
     i1 = (z8(:,i) - mu_t{2})'*inv(cov_t{2})*(z8(:,i)-mu_t{2})+log(det(cov_t{2}));
     i2 = (z8(:,i) - mu_t{3})'*inv(cov_t{3})*(z8(:,i)-mu_t{3})+log(det(cov_t{3}));
     i3 = (z8(:,i) - mu_t{4})'*inv(cov_t{4})*(z8(:,i)-mu_t{4})+log(det(cov_t{4}));
     i4 = (z8(:,i) - mu_t{5})'*inv(cov_t{5})*(z8(:,i)-mu_t{5})+log(det(cov_t{5}));
     i5 = (z8(:,i) - mu_t{6})'*inv(cov_t{6})*(z8(:,i)-mu_t{6})+log(det(cov_t{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 3
         ec3 = ec3 + 1;
     end
end

ec4 = 0;
for i = 1:10
     i0 = (z9(:,i) - mu_t{1})'*inv(cov_t{1})*(z9(:,i)-mu_t{1})+log(det(cov_t{1}));
     i1 = (z9(:,i) - mu_t{2})'*inv(cov_t{2})*(z9(:,i)-mu_t{2})+log(det(cov_t{2}));
     i2 = (z9(:,i) - mu_t{3})'*inv(cov_t{3})*(z9(:,i)-mu_t{3})+log(det(cov_t{3}));
     i3 = (z9(:,i) - mu_t{4})'*inv(cov_t{4})*(z9(:,i)-mu_t{4})+log(det(cov_t{4}));
     i4 = (z9(:,i) - mu_t{5})'*inv(cov_t{5})*(z9(:,i)-mu_t{5})+log(det(cov_t{5}));
     i5 = (z9(:,i) - mu_t{6})'*inv(cov_t{6})*(z9(:,i)-mu_t{6})+log(det(cov_t{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 4
         ec4 = ec4 + 1;
     end
end

ec5 = 0;
for i = 1:10
     i0 = (z10(:,i) - mu_t{1})'*inv(cov_t{1})*(z10(:,i)-mu_t{1})+log(det(cov_t{1}));
     i1 = (z10(:,i) - mu_t{2})'*inv(cov_t{2})*(z10(:,i)-mu_t{2})+log(det(cov_t{2}));
     i2 = (z10(:,i) - mu_t{3})'*inv(cov_t{3})*(z10(:,i)-mu_t{3})+log(det(cov_t{3}));
     i3 = (z10(:,i) - mu_t{4})'*inv(cov_t{4})*(z10(:,i)-mu_t{4})+log(det(cov_t{4}));
     i4 = (z10(:,i) - mu_t{5})'*inv(cov_t{5})*(z10(:,i)-mu_t{5})+log(det(cov_t{5}));
     i5 = (z10(:,i) - mu_t{6})'*inv(cov_t{6})*(z10(:,i)-mu_t{6})+log(det(cov_t{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 5
         ec5 = ec5 + 1;
     end
end

ec6 = 0;
for i = 1:10
     i0 = (z11(:,i) - mu_t{1})'*inv(cov_t{1})*(z11(:,i)-mu_t{1})+log(det(cov_t{1}));
     i1 = (z11(:,i) - mu_t{2})'*inv(cov_t{2})*(z11(:,i)-mu_t{2})+log(det(cov_t{2}));
     i2 = (z11(:,i) - mu_t{3})'*inv(cov_t{3})*(z11(:,i)-mu_t{3})+log(det(cov_t{3}));
     i3 = (z11(:,i) - mu_t{4})'*inv(cov_t{4})*(z11(:,i)-mu_t{4})+log(det(cov_t{4}));
     i4 = (z11(:,i) - mu_t{5})'*inv(cov_t{5})*(z11(:,i)-mu_t{5})+log(det(cov_t{5}));
     i5 = (z11(:,i) - mu_t{6})'*inv(cov_t{6})*(z11(:,i)-mu_t{6})+log(det(cov_t{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 6
         ec6 = ec6 + 1;
     end
end

total_error_pca = (ec1+ec2+ec3+ec4+ec5+ec6)/60;

clear i0 i1 i2 i3 i4 i5 i_s m i;

%% LDA Test Projection

z6 = w_t'*img6;
z7 = w_t'*img7;
z8 = w_t'*img8;
z9 = w_t'*img9;
z10 = w_t'*img10;
z11 = w_t'*img11;

%% LDA Projection Learning

z0 = w_t'*img0(:,1:40);
z1 = w_t'*img0(:,41:80);
z2 = w_t'*img0(:,81:120);
z3 = w_t'*img0(:,121:160);
z4 = w_t'*img0(:,161:200);
z5 = w_t'*img0(:,201:240);

z_mu0 = (1/40)*sum(z0,2);
z_mu1 = (1/40)*sum(z1,2);
z_mu2 = (1/40)*sum(z2,2);
z_mu3 = (1/40)*sum(z3,2);
z_mu4 = (1/40)*sum(z4,2);
z_mu5 = (1/40)*sum(z5,2);

z_cov0 = zeros(15, 15);
z_cov1 = zeros(15, 15);
z_cov2 = zeros(15, 15);
z_cov3 = zeros(15, 15);
z_cov4 = zeros(15, 15);
z_cov5 = zeros(15, 15);

for i = 1:40
    z_cov0 = z_cov0 + (z0(:,i) - z_mu0)*(z0(:,i) - z_mu0)';
end
z_cov0 = z_cov0/40;

for i = 1:40
    z_cov1 = z_cov1 + (z1(:,i) - z_mu1)*(z1(:,i) - z_mu1)';
end
z_cov1 = z_cov1/40;

for i = 1:40
    z_cov2 = z_cov2 + (z2(:,i) - z_mu2)*(z2(:,i) - z_mu2)';
end
z_cov2 = z_cov2/40;

for i = 1:40
    z_cov3 = z_cov3 + (z3(:,i) - z_mu3)*(z3(:,i) - z_mu3)';
end
z_cov3 = z_cov3/40;

for i = 1:40
    z_cov4 = z_cov4 + (z4(:,i) - z_mu4)*(z4(:,i) - z_mu4)';
end
z_cov4 = z_cov4/40;

for i = 1:40
    z_cov5 = z_cov5 + (z5(:,i) - z_mu5)*(z5(:,i) - z_mu5)';
end
z_cov5 = z_cov5/40;

mu_tl = {z_mu0,z_mu1,z_mu2,z_mu3,z_mu4,z_mu5};
cov_tl = {z_cov0,z_cov1,z_cov2,z_cov3,z_cov4,z_cov5};

clear z0 z1 z2 z3 z4 z5 z_mu0 z_mu1 z_mu2 z_mu3 z_mu4 z_mu5 ...
    z_cov0 z_cov1 z_cov2 z_cov3 z_cov4 z_cov5 i;

%% LDA Classification

c1l = 0;
for i = 1:10
     i0 = (z6(:,i) - mu_tl{1})'*inv(cov_tl{1})*(z6(:,i)-mu_tl{1})+log(det(cov_tl{1}));
     i1 = (z6(:,i) - mu_tl{2})'*inv(cov_tl{2})*(z6(:,i)-mu_tl{2})+log(det(cov_tl{2}));
     i2 = (z6(:,i) - mu_tl{3})'*inv(cov_tl{3})*(z6(:,i)-mu_tl{3})+log(det(cov_tl{3}));
     i3 = (z6(:,i) - mu_tl{4})'*inv(cov_tl{4})*(z6(:,i)-mu_tl{4})+log(det(cov_tl{4}));
     i4 = (z6(:,i) - mu_tl{5})'*inv(cov_tl{5})*(z6(:,i)-mu_tl{5})+log(det(cov_tl{5}));
     i5 = (z6(:,i) - mu_tl{6})'*inv(cov_tl{6})*(z6(:,i)-mu_tl{6})+log(det(cov_tl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 1
         c1l = c1l + 1;
     end
end

c2l = 0;
for i = 1:10
     i0 = (z7(:,i) - mu_tl{1})'*inv(cov_tl{1})*(z7(:,i)-mu_tl{1})+log(det(cov_tl{1}));
     i1 = (z7(:,i) - mu_tl{2})'*inv(cov_tl{2})*(z7(:,i)-mu_tl{2})+log(det(cov_tl{2}));
     i2 = (z7(:,i) - mu_tl{3})'*inv(cov_tl{3})*(z7(:,i)-mu_tl{3})+log(det(cov_tl{3}));
     i3 = (z7(:,i) - mu_tl{4})'*inv(cov_tl{4})*(z7(:,i)-mu_tl{4})+log(det(cov_tl{4}));
     i4 = (z7(:,i) - mu_tl{5})'*inv(cov_tl{5})*(z7(:,i)-mu_tl{5})+log(det(cov_tl{5}));
     i5 = (z7(:,i) - mu_tl{6})'*inv(cov_tl{6})*(z7(:,i)-mu_tl{6})+log(det(cov_tl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 2
         c2l = c2l + 1;
     end
end

c3l = 0;
for i = 1:10
     i0 = (z8(:,i) - mu_tl{1})'*inv(cov_tl{1})*(z8(:,i)-mu_tl{1})+log(det(cov_tl{1}));
     i1 = (z8(:,i) - mu_tl{2})'*inv(cov_tl{2})*(z8(:,i)-mu_tl{2})+log(det(cov_tl{2}));
     i2 = (z8(:,i) - mu_tl{3})'*inv(cov_tl{3})*(z8(:,i)-mu_tl{3})+log(det(cov_tl{3}));
     i3 = (z8(:,i) - mu_tl{4})'*inv(cov_tl{4})*(z8(:,i)-mu_tl{4})+log(det(cov_tl{4}));
     i4 = (z8(:,i) - mu_tl{5})'*inv(cov_tl{5})*(z8(:,i)-mu_tl{5})+log(det(cov_tl{5}));
     i5 = (z8(:,i) - mu_tl{6})'*inv(cov_tl{6})*(z8(:,i)-mu_tl{6})+log(det(cov_tl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 3
         c3l = c3l + 1;
     end
end

c4l = 0;
for i = 1:10
     i0 = (z9(:,i) - mu_tl{1})'*inv(cov_tl{1})*(z9(:,i)-mu_tl{1})+log(det(cov_tl{1}));
     i1 = (z9(:,i) - mu_tl{2})'*inv(cov_tl{2})*(z9(:,i)-mu_tl{2})+log(det(cov_tl{2}));
     i2 = (z9(:,i) - mu_tl{3})'*inv(cov_tl{3})*(z9(:,i)-mu_tl{3})+log(det(cov_tl{3}));
     i3 = (z9(:,i) - mu_tl{4})'*inv(cov_tl{4})*(z9(:,i)-mu_tl{4})+log(det(cov_tl{4}));
     i4 = (z9(:,i) - mu_tl{5})'*inv(cov_tl{5})*(z9(:,i)-mu_tl{5})+log(det(cov_tl{5}));
     i5 = (z9(:,i) - mu_tl{6})'*inv(cov_tl{6})*(z9(:,i)-mu_tl{6})+log(det(cov_tl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 4
         c4l = c4l + 1;
     end
end

c5l = 0;
for i = 1:10
     i0 = (z10(:,i) - mu_tl{1})'*inv(cov_tl{1})*(z10(:,i)-mu_tl{1})+log(det(cov_tl{1}));
     i1 = (z10(:,i) - mu_tl{2})'*inv(cov_tl{2})*(z10(:,i)-mu_tl{2})+log(det(cov_tl{2}));
     i2 = (z10(:,i) - mu_tl{3})'*inv(cov_tl{3})*(z10(:,i)-mu_tl{3})+log(det(cov_tl{3}));
     i3 = (z10(:,i) - mu_tl{4})'*inv(cov_tl{4})*(z10(:,i)-mu_tl{4})+log(det(cov_tl{4}));
     i4 = (z10(:,i) - mu_tl{5})'*inv(cov_tl{5})*(z10(:,i)-mu_tl{5})+log(det(cov_tl{5}));
     i5 = (z10(:,i) - mu_tl{6})'*inv(cov_tl{6})*(z10(:,i)-mu_tl{6})+log(det(cov_tl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 5
         c5l = c5l + 1;
     end
end

c6l = 0;
for i = 1:10
     i0 = (z11(:,i) - mu_tl{1})'*inv(cov_tl{1})*(z11(:,i)-mu_tl{1})+log(det(cov_tl{1}));
     i1 = (z11(:,i) - mu_tl{2})'*inv(cov_tl{2})*(z11(:,i)-mu_tl{2})+log(det(cov_tl{2}));
     i2 = (z11(:,i) - mu_tl{3})'*inv(cov_tl{3})*(z11(:,i)-mu_tl{3})+log(det(cov_tl{3}));
     i3 = (z11(:,i) - mu_tl{4})'*inv(cov_tl{4})*(z11(:,i)-mu_tl{4})+log(det(cov_tl{4}));
     i4 = (z11(:,i) - mu_tl{5})'*inv(cov_tl{5})*(z11(:,i)-mu_tl{5})+log(det(cov_tl{5}));
     i5 = (z11(:,i) - mu_tl{6})'*inv(cov_tl{6})*(z11(:,i)-mu_tl{6})+log(det(cov_tl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 6
         c6l = c6l + 1;
     end
end

total_error_lda = (c1l+c2l+c3l+c4l+c5l+c6l)/60;

clear i0 i1 i2 i3 i4 i5 i_s m i;

%% PCD + LDA Training Projection

z0 = pc_16'*img0(:,1:40);
z1 = pc_16'*img0(:,41:80);
z2 = pc_16'*img0(:,81:120);
z3 = pc_16'*img0(:,121:160);
z4 = pc_16'*img0(:,161:200);
z5 = pc_16'*img0(:,201:240);

mu0 = (1/40)*sum(z0,2);
mu1 = (1/40)*sum(z1,2);
mu2 = (1/40)*sum(z2,2);
mu3 = (1/40)*sum(z3,2);
mu4 = (1/40)*sum(z4,2);
mu5 = (1/40)*sum(z5,2);

cov0 = zeros(30,30);
cov1 = zeros(30,30);
cov2 = zeros(30,30);
cov3 = zeros(30,30);
cov4 = zeros(30,30);
cov5 = zeros(30,30);

for i = 1:40
    cov0 = cov0 + (z0(:,i)-mu0)*(z0(:,i)-mu0)';
end
cov0 = cov0/40;

for i = 1:40
    cov1 = cov1 + (z1(:,i)-mu1)*(z1(:,i)-mu1)';
end
cov1 = cov1/40;

for i = 1:40
    cov2 = cov2 + (z2(:,i)-mu2)*(z2(:,i)-mu2)';
end
cov2 = cov2/40;

for i = 1:40
    cov3 = cov3 + (z3(:,i)-mu3)*(z3(:,i)-mu3)';
end
cov3 = cov3/40;

for i = 1:40
    cov4 = cov4 + (z4(:,i)-mu4)*(z4(:,i)-mu4)';
end
cov4 = cov4/40;

for i = 1:40
    cov5 = cov5 + (z5(:,i)-mu5)*(z5(:,i)-mu5)';
end
cov5 = cov5/40;

w01 = (cov0 + cov1)\(mu0 - mu1);
w02 = (cov0 + cov2)\(mu0 - mu2);
w03 = (cov0 + cov3)\(mu0 - mu3);
w04 = (cov0 + cov4)\(mu0 - mu4);
w05 = (cov0 + cov5)\(mu0 - mu5);
w12 = (cov1 + cov2)\(mu1 - mu2);
w13 = (cov1 + cov3)\(mu1 - mu3);
w14 = (cov1 + cov4)\(mu1 - mu4);
w15 = (cov1 + cov5)\(mu1 - mu5);
w23 = (cov2 + cov3)\(mu2 - mu3);
w24 = (cov2 + cov4)\(mu2 - mu4);
w25 = (cov2 + cov5)\(mu2 - mu5);
w34 = (cov3 + cov4)\(mu3 - mu4);
w35 = (cov3 + cov5)\(mu3 - mu5);
w45 = (cov4 + cov5)\(mu4 - mu5);

w_t2 = zeros(30, 15);
w_t2(:,1) = w01;
w_t2(:,2) = w02;
w_t2(:,3) = w03;
w_t2(:,4) = w04;
w_t2(:,5) = w05;
w_t2(:,6) = w12;
w_t2(:,7) = w13;
w_t2(:,8) = w14;
w_t2(:,9) = w15;
w_t2(:,10) = w23;
w_t2(:,11) = w24;
w_t2(:,12) = w25;
w_t2(:,13) = w34;
w_t2(:,14) = w35;
w_t2(:,15) = w45;

clear mu0 mu1 mu2 mu3 mu4 mu5 cov0 cov1 cov2 cov3 cov4 cov5 i ...
    w01 w02 w03 w04 w05 w12 w13 w14 w15 w23 w24 w25 w34 w35 w45;

%% PCA + LDA Training Learning

z0 = pc_16'*img0(:,1:40);
z1 = pc_16'*img0(:,41:80);
z2 = pc_16'*img0(:,81:120);
z3 = pc_16'*img0(:,121:160);
z4 = pc_16'*img0(:,161:200);
z5 = pc_16'*img0(:,201:240);

z0 = w_t2'*z0;
z1 = w_t2'*z1;
z2 = w_t2'*z2;
z3 = w_t2'*z3;
z4 = w_t2'*z4;
z5 = w_t2'*z5;

z_mu0 = (1/40)*sum(z0,2);
z_mu1 = (1/40)*sum(z1,2);
z_mu2 = (1/40)*sum(z2,2);
z_mu3 = (1/40)*sum(z3,2);
z_mu4 = (1/40)*sum(z4,2);
z_mu5 = (1/40)*sum(z5,2);

z_cov0 = zeros(15, 15);
z_cov1 = zeros(15, 15);
z_cov2 = zeros(15, 15);
z_cov3 = zeros(15, 15);
z_cov4 = zeros(15, 15);
z_cov5 = zeros(15, 15);

for i = 1:40
    z_cov0 = z_cov0 + (z0(:,i) - z_mu0)*(z0(:,i) - z_mu0)';
end
z_cov0 = z_cov0/40;

for i = 1:40
    z_cov1 = z_cov1 + (z1(:,i) - z_mu1)*(z1(:,i) - z_mu1)';
end
z_cov1 = z_cov1/40;

for i = 1:40
    z_cov2 = z_cov2 + (z2(:,i) - z_mu2)*(z2(:,i) - z_mu2)';
end
z_cov2 = z_cov2/40;

for i = 1:40
    z_cov3 = z_cov3 + (z3(:,i) - z_mu3)*(z3(:,i) - z_mu3)';
end
z_cov3 = z_cov3/40;

for i = 1:40
    z_cov4 = z_cov4 + (z4(:,i) - z_mu4)*(z4(:,i) - z_mu4)';
end
z_cov4 = z_cov4/40;

for i = 1:40
    z_cov5 = z_cov5 + (z5(:,i) - z_mu5)*(z5(:,i) - z_mu5)';
end
z_cov5 = z_cov5/40;

mu_tpl = {z_mu0,z_mu1,z_mu2,z_mu3,z_mu4,z_mu5};
cov_tpl = {z_cov0,z_cov1,z_cov2,z_cov3,z_cov4,z_cov5};

clear z0 z1 z2 z3 z4 z5 z_mu0 z_mu1 z_mu2 z_mu3 z_mu4 z_mu5 ...
    z_cov0 z_cov1 z_cov2 z_cov3 z_cov4 z_cov5 i;

%% PCA + LDA Test Projection

z6 = pc_16'*img6;
z7 = pc_16'*img7;
z8 = pc_16'*img8;
z9 = pc_16'*img9;
z10 = pc_16'*img10;
z11 = pc_16'*img11;

z6 = w_t2'*z6;
z7 = w_t2'*z7;
z8 = w_t2'*z8;
z9 = w_t2'*z9;
z10 = w_t2'*z10;
z11 = w_t2'*z11;

%% PCA + LDA Classification

c1pl = 0;
for i = 1:10
     i0 = (z6(:,i) - mu_tpl{1})'*inv(cov_tpl{1})*(z6(:,i)-mu_tpl{1})+log(det(cov_tpl{1}));
     i1 = (z6(:,i) - mu_tpl{2})'*inv(cov_tpl{2})*(z6(:,i)-mu_tpl{2})+log(det(cov_tpl{2}));
     i2 = (z6(:,i) - mu_tpl{3})'*inv(cov_tpl{3})*(z6(:,i)-mu_tpl{3})+log(det(cov_tpl{3}));
     i3 = (z6(:,i) - mu_tpl{4})'*inv(cov_tpl{4})*(z6(:,i)-mu_tpl{4})+log(det(cov_tpl{4}));
     i4 = (z6(:,i) - mu_tpl{5})'*inv(cov_tpl{5})*(z6(:,i)-mu_tpl{5})+log(det(cov_tpl{5}));
     i5 = (z6(:,i) - mu_tpl{6})'*inv(cov_tpl{6})*(z6(:,i)-mu_tpl{6})+log(det(cov_tpl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 1
         c1pl = c1pl + 1;
     end
end

c2pl = 0;
for i = 1:10
     i0 = (z7(:,i) - mu_tpl{1})'*inv(cov_tpl{1})*(z7(:,i)-mu_tpl{1})+log(det(cov_tpl{1}));
     i1 = (z7(:,i) - mu_tpl{2})'*inv(cov_tpl{2})*(z7(:,i)-mu_tpl{2})+log(det(cov_tpl{2}));
     i2 = (z7(:,i) - mu_tpl{3})'*inv(cov_tpl{3})*(z7(:,i)-mu_tpl{3})+log(det(cov_tpl{3}));
     i3 = (z7(:,i) - mu_tpl{4})'*inv(cov_tpl{4})*(z7(:,i)-mu_tpl{4})+log(det(cov_tpl{4}));
     i4 = (z7(:,i) - mu_tpl{5})'*inv(cov_tpl{5})*(z7(:,i)-mu_tpl{5})+log(det(cov_tpl{5}));
     i5 = (z7(:,i) - mu_tpl{6})'*inv(cov_tpl{6})*(z7(:,i)-mu_tpl{6})+log(det(cov_tpl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 2
         c2pl = c2pl + 1;
     end
end

c3pl = 0;
for i = 1:10
     i0 = (z8(:,i) - mu_tpl{1})'*inv(cov_tpl{1})*(z8(:,i)-mu_tpl{1})+log(det(cov_tpl{1}));
     i1 = (z8(:,i) - mu_tpl{2})'*inv(cov_tpl{2})*(z8(:,i)-mu_tpl{2})+log(det(cov_tpl{2}));
     i2 = (z8(:,i) - mu_tpl{3})'*inv(cov_tpl{3})*(z8(:,i)-mu_tpl{3})+log(det(cov_tpl{3}));
     i3 = (z8(:,i) - mu_tpl{4})'*inv(cov_tpl{4})*(z8(:,i)-mu_tpl{4})+log(det(cov_tpl{4}));
     i4 = (z8(:,i) - mu_tpl{5})'*inv(cov_tpl{5})*(z8(:,i)-mu_tpl{5})+log(det(cov_tpl{5}));
     i5 = (z8(:,i) - mu_tpl{6})'*inv(cov_tpl{6})*(z8(:,i)-mu_tpl{6})+log(det(cov_tpl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 3
         c3pl = c3pl + 1;
     end
end

c4pl = 0;
for i = 1:10
     i0 = (z9(:,i) - mu_tpl{1})'*inv(cov_tpl{1})*(z9(:,i)-mu_tpl{1})+log(det(cov_tpl{1}));
     i1 = (z9(:,i) - mu_tpl{2})'*inv(cov_tpl{2})*(z9(:,i)-mu_tpl{2})+log(det(cov_tpl{2}));
     i2 = (z9(:,i) - mu_tpl{3})'*inv(cov_tpl{3})*(z9(:,i)-mu_tpl{3})+log(det(cov_tpl{3}));
     i3 = (z9(:,i) - mu_tpl{4})'*inv(cov_tpl{4})*(z9(:,i)-mu_tpl{4})+log(det(cov_tpl{4}));
     i4 = (z9(:,i) - mu_tpl{5})'*inv(cov_tpl{5})*(z9(:,i)-mu_tpl{5})+log(det(cov_tpl{5}));
     i5 = (z9(:,i) - mu_tpl{6})'*inv(cov_tpl{6})*(z9(:,i)-mu_tpl{6})+log(det(cov_tpl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 4
         c4pl = c4pl + 1;
     end
end

c5pl = 0;
for i = 1:10
     i0 = (z10(:,i) - mu_tpl{1})'*inv(cov_tpl{1})*(z10(:,i)-mu_tpl{1})+log(det(cov_tpl{1}));
     i1 = (z10(:,i) - mu_tpl{2})'*inv(cov_tpl{2})*(z10(:,i)-mu_tpl{2})+log(det(cov_tpl{2}));
     i2 = (z10(:,i) - mu_tpl{3})'*inv(cov_tpl{3})*(z10(:,i)-mu_tpl{3})+log(det(cov_tpl{3}));
     i3 = (z10(:,i) - mu_tpl{4})'*inv(cov_tpl{4})*(z10(:,i)-mu_tpl{4})+log(det(cov_tpl{4}));
     i4 = (z10(:,i) - mu_tpl{5})'*inv(cov_tpl{5})*(z10(:,i)-mu_tpl{5})+log(det(cov_tpl{5}));
     i5 = (z10(:,i) - mu_tpl{6})'*inv(cov_tpl{6})*(z10(:,i)-mu_tpl{6})+log(det(cov_tpl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 5
         c5pl = c5pl + 1;
     end
end

c6pl = 0;
for i = 1:10
     i0 = (z11(:,i) - mu_tpl{1})'*inv(cov_tpl{1})*(z11(:,i)-mu_tpl{1})+log(det(cov_tpl{1}));
     i1 = (z11(:,i) - mu_tpl{2})'*inv(cov_tpl{2})*(z11(:,i)-mu_tpl{2})+log(det(cov_tpl{2}));
     i2 = (z11(:,i) - mu_tpl{3})'*inv(cov_tpl{3})*(z11(:,i)-mu_tpl{3})+log(det(cov_tpl{3}));
     i3 = (z11(:,i) - mu_tpl{4})'*inv(cov_tpl{4})*(z11(:,i)-mu_tpl{4})+log(det(cov_tpl{4}));
     i4 = (z11(:,i) - mu_tpl{5})'*inv(cov_tpl{5})*(z11(:,i)-mu_tpl{5})+log(det(cov_tpl{5}));
     i5 = (z11(:,i) - mu_tpl{6})'*inv(cov_tpl{6})*(z11(:,i)-mu_tpl{6})+log(det(cov_tpl{6}));
     i_s = [i0, i1, i2, i3, i4, i5];
     [~,m] = min(i_s);
     if m ~= 6
         c6pl = c6pl + 1;
     end
end

total_error_pcalda = (c1pl+c2pl+c3pl+c4pl+c5pl+c6pl)/60;

clear i0 i1 i2 i3 i4 i5 i_s m i;
