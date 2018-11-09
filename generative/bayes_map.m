%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Homework 3/4
% Problem 4
% ECE271A - Statistical Learning I
% Ibrahim Akbar
% 18/11/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% VARIABLE LOADING

clear;
load('TrainingSamplesDCT_subsets_8.mat');
load('Prior_1.mat');
load('Alpha.mat');

% Creating prior parameters
sigma_null = zeros(64, 64);

for i = 1:64
    sigma_null(i, i) = W0(i);
end

mu0_FG = mu0_FG';
mu0_BG = mu0_BG';

sizes = zeros(4,2);
L_bg = size(D1_BG);
L_fg = size(D1_FG);
sizes(1,1) = L_fg(1);
sizes(1,2) = L_bg(1);
L_bg = size(D2_BG);
L_fg = size(D2_FG);
sizes(2,1) = L_fg(1);
sizes(2,2) = L_bg(1);
L_bg = size(D3_BG);
L_fg = size(D3_FG);
sizes(3,1) = L_fg(1);
sizes(3,2) = L_bg(1);
L_bg = size(D4_BG);
L_fg = size(D4_FG);
sizes(4,1) = L_fg(1);
sizes(4,2) = L_bg(1);

train_data = cell(4,2);
train_data{1,1} = D1_FG;
train_data{1,2} = D1_BG;
train_data{2,1} = D2_FG;
train_data{2,2} = D2_BG;
train_data{3,1} = D3_FG;
train_data{3,2} = D3_BG;
train_data{4,1} = D4_FG;
train_data{4,2} = D4_BG;

clear i D1_FG D1_BG D2_FG D2_BG D3_FG D3_BG D4_FG D4_BG L_fg L_bg;
%% PRIOR CLASS PARAMETERS

L_bg = sizes(1,1);
L_fg = sizes(1,2);

bg_prior = L_bg/(L_bg+L_fg);
fg_prior = L_fg/(L_bg+L_fg);

clear L_bg L_fg;
%% ML Parameters
fg_mu_ml = zeros(64,1);
bg_mu_ml = zeros(64,1);

fg_sigma_ml = zeros(64,64);
bg_sigma_ml = zeros(64,64);

L = size(sizes);

ml_mu = cell(4,2);
ml_sigma = cell(4,2);

for k = 1:L(1)
    
    D1_FG = train_data{k,1};
    D1_BG = train_data{k,2};
    
    for i = 1:64
        fg_mu_ml(i) = (1/sizes(k,1))*sum(D1_FG(:,i));
        bg_mu_ml(i) = (1/sizes(k,2))*sum(D1_BG(:,i));
    end

    ml_mu{k,1} = fg_mu_ml;
    ml_mu{k,2} = bg_mu_ml;

    for i = 2:64
        for j = i:64
            sigmas_fg = cov(D1_FG(:,i-1),D1_FG(:,j));
            sigmas_bg = cov(D1_BG(:,i-1),D1_BG(:,j));

            fg_sigma_ml(i-1,i-1) = sigmas_fg(1,1);
            fg_sigma_ml(i-1,j) = sigmas_fg(1,2);
            fg_sigma_ml(j,i-1) = sigmas_fg(2,1);
            fg_sigma_ml(j,j) = sigmas_fg(2,2);

            bg_sigma_ml(i-1,i-1) = sigmas_bg(1,1);
            bg_sigma_ml(i-1,j) = sigmas_bg(1,2);
            bg_sigma_ml(j,i-1) = sigmas_bg(2,1);
            bg_sigma_ml(j,j) = sigmas_bg(2,2);
        end
    end

    ml_sigma{k,1} = fg_sigma_ml;
    ml_sigma{k,2} = bg_sigma_ml;

    fg_mu_ml = zeros(64,1);
    bg_mu_ml = zeros(64,1);

    fg_sigma_ml = zeros(64,64);
    bg_sigma_ml = zeros(64,64);
end

clear fg_sigma_ml bg_sigma_ml fg_mu_ml bg_mu_ml i j k L L_bg L_fg sigmas_bg sigmas_fg D1_FG D1_BG;
%% MAP Parameters

map_mu = cell(4,2);
map_sigma = cell(4,2);

L = size(sizes);

for i = 1:L(1)
    fg_mu_ml = ml_mu{i,1};
    bg_mu_ml = ml_mu{i,2};

    fg_sigma_ml = ml_sigma{i,1};
    bg_sigma_ml = ml_sigma{i,2};

    L_bg = sizes(i,2);
    L_fg = sizes(i,1);

    fg_sigma_n = inv(L_fg(1)*inv(fg_sigma_ml)+inv(sigma_null));
    bg_sigma_n = inv(L_bg(1)*inv(bg_sigma_ml)+inv(sigma_null));

    fg_mu_n = fg_sigma_n*((L_fg(1)*inv(fg_sigma_ml))*fg_mu_ml+inv(sigma_null)*mu0_FG);
    bg_mu_n = bg_sigma_n*((L_bg(1)*inv(bg_sigma_ml))*bg_mu_ml+inv(sigma_null)*mu0_BG);

    map_mu{i,1} = fg_mu_n;
    map_mu{i,2} = bg_mu_n;

    map_sigma{i,1} = fg_sigma_n;
    map_sigma{i,2} = bg_sigma_n;
end

clear fg_sigma_n bg_sigma_n bg_mu_ml fg_mu_ml fg_mu_n bg_mu_n L_bg L_fg fg_sigma_ml bg_sigma_ml i L;
%% Alpha Mu Computations

L = length(alpha);
fg_alpha_mu_n = cell(L,1);
bg_alpha_mu_n = cell(L,1);

alpha_mu = cell(4,2);

sig_null = cell(L,1);

flag = 0;

for j = 1:4
    for i = 1:L
       fg_mu = alpha(i)*ml_mu{j,1}+(1-alpha(i))*mu0_FG;
       bg_mu = alpha(i)*ml_mu{j,2}+(1-alpha(i))*mu0_BG;

       fg_alpha_mu_n{i} = fg_mu;
       bg_alpha_mu_n{i} = bg_mu;
       
       if(flag == 0)
           sig_null{i} = alpha(i)*sigma_null;
           if(i == L)
               flag = 1;
           end
       end
    end
    alpha_mu{j,1} = fg_alpha_mu_n;
    alpha_mu{j,2} = bg_alpha_mu_n;
    
    fg_alpha_mu_n = cell(L,1);
    bg_alpha_mu_n = cell(L,1);
end

clear i j k L fg_mu bg_mu flag fg_alpha_mu_n bg_alpha_mu_n;
%% Alpha Sigma Computations

L = length(alpha);

fg_alpha_sigma_n = cell(L,1);
bg_alpha_sigma_n = cell(L,1);

alpha_sigma = cell(4,2);

for i = 1:4
   
    fg_sigma_ml = ml_sigma{i,1};
    bg_sigma_ml = ml_sigma{i,2};

    L_bg = sizes(i,2);
    L_fg = sizes(i,1);
    
    for j = 1:L

        fg_sigma_n = inv(L_fg(1)*inv(fg_sigma_ml)+inv(sig_null{j}));
        bg_sigma_n = inv(L_bg(1)*inv(bg_sigma_ml)+inv(sig_null{j}));
        
        fg_alpha_sigma_n{j} = fg_sigma_n;
        bg_alpha_sigma_n{j} = fg_sigma_n;
        
    end
    
    alpha_sigma{i,1} = fg_alpha_sigma_n;
    alpha_sigma{i,2} = bg_alpha_sigma_n;
    
    fg_alpha_sigma_n = cell(L,1);
    bg_alpha_sigma_n = cell(L,1);
    
end

clear bg_alpha_sigma_n fg_alpha_sigma_n bg_sigma_ml fg_sigma_ml fg_sigma_n bg_sigma_n i j L L_bg L_fg;
%% Image Formatting

% Data Preparation
imagefile = './images/barbara.bmp';
filename = 'Zig-Zag Pattern.txt';

zigFile = fopen(filename, 'r');
zigzag = fscanf(zigFile, '%d');
fclose(zigFile);

zigzag = zigzag + ones(length(zigzag),1);

image = imread(imagefile);
figure;
imagesc(image);
image = im2double(image);

% Padding may not be necessary
% image = [zeros(255,1), image, zeros(255,1)];
% image = [zeros(1,272); image; zeros(1,272)];
% image = [image,zeros(255,1)];
% image = [image;zeros(1,271)];

S = size(image);

% Windowing of the image
blocks = cell(S(1)-7,S(2)-7);

for i = 8:S(1)
    for j = 8:S(2)
        blocks{i-7,j-7} = image(i-7:i,j-7:j);
    end
end

% Discrete Cosine Transform and Vector Space Adjustment
S = size(blocks);
features = zeros(64, S(1)*S(2));
vectorBlock = zeros(length(zigzag),1);
extender = 0;

for i = 1:S(1)
    for j = 1:S(2)
        block = blocks{i,j};
        block = dct2(block);
        vectorBlock(zigzag) = block;
        features(:, j+extender) = vectorBlock;
    end
    extender = extender + S(2);
end

clear extender block vectorBlock i j blocks zigzag imagefile filename zigFile image ans;
%% BDR per Pixel

L = length(features);

As = cell(1,4);

for i = 1:4
    
    A = zeros(L,1);
    
    fg_mu = ml_mu{i,1};
    bg_mu = ml_mu{i,2};
    
    fg_sigma = map_sigma{i,1};
    bg_sigma = map_sigma{i,2};
    
    fg_sigma_ml = ml_sigma{i,1};
    bg_sigma_ml = ml_sigma{i,2};
    
    fg_sigma = fg_sigma+fg_sigma_ml;
    bg_sigma = bg_sigma+bg_sigma_ml;
    
    fg_term_2 = log((2*pi)^(64)*det(fg_sigma));
    bg_term_2 = log((2*pi)^(64)*det(bg_sigma));
    
    for j = 1:L
        
        fg_term_1 = (features(j,:)'-fg_mu)'/fg_sigma;
        fg_term_1 = fg_term_1*(features(j,:)'-fg_mu);
        
        bg_term_1 = (features(j,:)'-bg_mu)'/bg_sigma;
        bg_term_1 = bg_term_1*(features(j,:)'-bg_mu);
        
        i_fg = fg_term_1+fg_term_2-2*log(fg_prior);
        i_bg = bg_term_1+bg_term_2-2*log(bg_prior);
        
        if(i_bg > i_fg)
            A(j) = 1;
        else
            A(j) = 0;
        end  
    end
    
    As{i} = A;
end

images = cell(4,1);
image_D1 = zeros(S(1)+7,S(2)+7);
incrementer = 1;

for k = 1:4
    A = As{k};
    for i = 8:S(1)+7
        for j = 8:S(2)+7
            image_D1(i-7:i,j-7:j) = A(incrementer);
            incrementer = incrementer + 1;
        end
    end
    images{k} = image_D1;
    image_D1 = zeros(S(1)+7,S(2)+7);
    incrementer = 1;
end

clear i j k incrementer image_D1 A L fg_mu bg_mu fg_sigma bg_sigma fg_sigma_ml bg_sigma_ml fg_term_1 bg_term_1 ...
    fg_term_2 bg_term_2 i_bg i_fg ans;
%% BDR with Alpha Parameters

L = length(features);
masks = cell(4,1);

for i = 1:4
    
%     fg_a_mu = alpha_mu{i,1};
%     bg_a_mu = alpha_mu{i,2};
    
    fg_sigma_ml = ml_sigma{i,1};
    bg_sigma_ml = ml_sigma{i,2};
    
%     fg_a_sigma = alpha_sigma{i,1};
%     bg_a_sigma = alpha_sigma{i,2};
    
    mask = cell(9,1);
    
    for j = 1:9
        
        A = zeros(L,1);
        
%         fg_mu = fg_a_mu{j};
%         bg_mu = bg_a_mu{j};
        
        fg_mu = ml_mu{i,1};
        bg_mu = ml_mu{i,2};
        
%         fg_sigma = fg_a_sigma{j};
%         bg_sigma = bg_a_sigma{j};
        
%         fg_sigma = fg_sigma+fg_sigma_ml;
%         bg_sigma = bg_sigma+bg_sigma_ml;
         
        fg_sigma = fg_sigma_ml;
        bg_sigma = bg_sigma_ml;
        
        fg_term_2 = log((2*pi)^(64)*det(fg_sigma));
        bg_term_2 = log((2*pi)^(64)*det(bg_sigma));
        
        for k = 1:L
            
            fg_term_1 = (features(k,:)'-fg_mu)'/fg_sigma;
            fg_term_1 = fg_term_1*(features(k,:)'-fg_mu);
        
            bg_term_1 = (features(k,:)'-bg_mu)'/bg_sigma;
            bg_term_1 = bg_term_1*(features(k,:)'-bg_mu);
        
            i_fg = fg_term_1+fg_term_2-2*log(fg_prior);
            i_bg = bg_term_1+bg_term_2-2*log(bg_prior);
            
            if(i_bg > i_fg)
                A(k) = 1;
            else
                A(k) = 0;
            end
        end
        mask{j} = A;
        
    end
    masks{i} = mask;
end

clear i j k fg_a_mu bg_a_mu fg_a_sigma bg_a_sigma fg_sigma_ml bg_sigma_ml mask A...
    fg_mu bg_mu fg_sigma bg_sigma A fg_term_1 bg_term_1 fg_term_2 bg_term_2 i_fg i_bg L;
%% Image Restoration

images = cell(4,1);
image_D1 = zeros(S(1)+7,S(2)+7);
incrementer = 1;

for k = 1:4
    mask = masks{k};
    image_alpha = cell(length(mask),1);
    
    for l = 1:length(mask)
        A = mask{l};
        
        for i = 8:S(1)+7
            for j = 8:S(2)+7
                image_D1(i-7:i,j-7:j) = A(incrementer);
                incrementer = incrementer + 1;
            end
        end
        image_alpha{l} = image_D1;
        image_D1 = zeros(S(1)+7,S(2)+7);
        incrementer = 1;
    end
    images{k} = image_alpha;
end

clear i j l k incrementer image_alpha image_D1 A mask S;
%% Grey scale

for i = 1:4
    image_alpha = images{i};
    
    for j = 1:9
        image = image_alpha{j};
        
        figure;
        imagesc(image);
        colormap(gray(255));
    end
end

clear i j image_alpha image;
%% Probability of Error

% Computing the error rate from the ground truth
% Error Rate = # Incorrect/# Total Pixels
imagefile = 'cheetah_mask.bmp';
figure;
truth = imread(imagefile);
imagesc(truth);
truth = im2double(truth);

S = size(truth);
fg_size = sum(truth);
fg_size = sum(fg_size);

probability = cell(4,1);

for k = 1:4
    
    image_alpha = images{k};
    probs = zeros(9,1);
    
    for l = 1:9
        
        image = image_alpha{l};
        
        detect = 0;
        error = 0;
        
        for i = 1:S(1)
            for j = 1:S(2)
                if((truth(i,j) == image(i,j)) && truth(i,j) == 1)
                    detect = detect + 1;
                end
                if(truth(i,j) ~= image(i,j) && truth(i,j) == 0)
                    error = error + 1;
                end
            end
        end
        
        p_detect = detect/fg_size;
        p_error = error/(S(1)*S(2)-fg_size);
        poe = (1-p_detect)*fg_prior+p_error*bg_prior;
        probs(l) = poe;
    end
    
    probability{k} = probs;
end


clear k l i j imagefile S fg_size p_detect p_error error detect poe probs image_alpha image L;

%% PoE Plots

for i = 1:4
    
    probs = probability{i};
    
    figure;
    semilogx(alpha, probs);
    xlabel('Alphas');
    ylabel('Probability of Error');
    ylim([0 1]);
    title(['PoE vs. Alpha D:',num2str(i)]);
    
end

clear i probs;