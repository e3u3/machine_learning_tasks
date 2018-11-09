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

%% Data Points

clear image;
classes = cell(10,1);
for i = 1:10
    indices = find(train_labels == i-1);
    class = train_images(indices,:);
    classes{i,1} = class;
    if(i-1 == 5)
        sample = randi(length(indices));
        point = class(sample,:);
        image = reshape(point, [28,28]);
        image = image';
    end
end

images = cell(9,1);
figure(1);
for i = 1:9
   template = zeros(70,70);
   template(5*(i-1)+1:5*(i-1)+28,21:48) = image(:,:);
   subplot(3,3,i);
   imagesc(template);
   images{i,1} = template;
end

colormap(gray);

figure(2);
imagesc(image);
title('Kernel for "5" Detection');
colormap(gray);

%% Encoder

kernel = zeros(70,70);
kernel2 = zeros(70,70);
kernel2(35,35) = 1;
kernel(20:47,21:48) = image;
A = zeros(4900,4900);
A2 = zeros(4900,4900);
new_img = kernel;
new_img2 = kernel2;
for i = 1:4900
   img = reshape(new_img, [4900,1]);
   img2 = reshape(new_img2, [4900,1]);
   A(i,:) = img;
   A2(i,:) = img2;
   new_img = circshift(new_img,1,1);
   new_img2 = circshift(new_img2,1,1);
   if(mod(i,70) == 0)
      new_img = circshift(new_img,1,2);
      new_img2 = circshift(new_img2,1,2);
   end
end
figure(3);
imagesc(A);

colormap(gray);

%% Convolution Output

figure(5);
imagesc(conv2(image,image));
colormap(gray);

convs = cell(9,2);
figure(6);
for i = 1:9
   subplot(3,3,i);
   input = images{i};
   input = reshape(input, [4900,1]);
   conv = A*input;
   conv2i = A2*input;
   conv = reshape(conv, [70,70]);
   conv2i = reshape(conv2i, [70,70]);
   convs{i,1} = conv;
   convs{i,2} = conv2i;
   imagesc(conv);
end
colormap(gray);

figure(7);
for i = 1:9
    subplot(3,3,i);
    conv = convs{i};
    conv = circshift(conv,35,1);
    conv = circshift(conv,35,2);
    imagesc(conv);
end
colormap(gray);

%% Decoder

figure(8);
for i = 1:9
    subplot(3,3,i);
    conv2m = convs{i,1};
    conv2m = reshape(conv2m, [4900, 1]);
    recon = A'*conv2m;
    recon = reshape(recon, [70,70]);
    imagesc(recon);
end
colormap(gray);

figure(9);
for i = 1:9
    subplot(3,3,i);
    conv2m = convs{i,2};
    conv2m = reshape(conv2m, [4900, 1]);
    recon = A2'*conv2m;
    recon = reshape(recon, [70,70]);
    imagesc(recon);
end
colormap(gray);

%% Kernel Maker

kerns = cell(10,1);
ones = zeros(28,1);
for i = 1:10
    file = sprintf('./hw3p6data/%d.png',i-1);
    img = imread(file);
    img = double(img);
    mu = (1/28)*img*ones;
    for j = 1:28
        img(:,j) = img(:,j)-mu;
    end
    kernel = zeros(120,200);
    kernel(46:73,86:113) = img;
    kerns{i,1} = kernel;
end

%% Convolutions Encoder Maker

file = './hw3p6data/img.png';
test = imread(file);
test = double(test);

As = cell(10,1);
for i = 1:10
    A = zeros(24000, 24000);
    kernel = kerns{i,1};
    new_img = kernel;
    for j = 1:24000
       img = reshape(new_img, [24000,1]);
       A(i,:) = img;
       new_img = circshift(new_img,1,1);
       if(mod(i,120) == 0)
          new_img = circshift(new_img,1,2);
       end
    end
    As{i,1} = A;
end

%% Encoding

codes = zeros(24000, 10);
test = reshape(test, [24000, 1]);

figure(12);
for i = 1:10
    subplot(5,2,i);
    A = As{i,1};
    code = A*test;
    codes(:,1) = code;
    code = reshape(code, [120,200]);
    imagesc(code);
end