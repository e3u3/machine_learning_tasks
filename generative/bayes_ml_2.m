%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Homework 2
% Problem 6
% ECE271A - Statistical Learning I
% Ibrahim Akbar
% 20/10/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Part A
% Finding the prior probabilities:
% P_y(cheetah) & P_y(grass)
% Idea_1: Using the sizes of the given matrices determine the
% percentage of these matrices are of the total matrix (i.e. the whole
% image), Thus giving a probability of the priors.
%
% Think of other ways....Go to office hours and inquiry about your ideas.

%NOTES
% each row -> zig zag scanned vector of coefficients
clear;
load('TrainingSamplesDCT_8_new.mat');

L = [length(TrainsampleDCT_FG),length(TrainsampleDCT_BG)];
prior_cheetah = L(1)/(L(1)+L(2));
prior_grass = L(2)/(L(1)+L(2));

%% Part B_1
% Goal: Determine mu & covariance of multivariate gaussian
mu_cheetah = zeros(64,1);
mu_grass = zeros(64,1);

covar_cheetah = zeros(64,64);
covar_grass = zeros(64,64);

for i = 1:64
    mu_cheetah(i) = (1/L(1))*sum(TrainsampleDCT_FG(:,i));
    mu_grass(i) = (1/L(2))*sum(TrainsampleDCT_BG(:,i));
end

for i = 2:64
    for j = i:64
        sigmas_cheetah = cov(TrainsampleDCT_FG(:,i-1),TrainsampleDCT_FG(:,j));
        sigmas_grass = cov(TrainsampleDCT_BG(:,i-1),TrainsampleDCT_BG(:,j));
        
        covar_cheetah(i-1,i-1) = sigmas_cheetah(1,1);
        covar_cheetah(i-1,j) = sigmas_cheetah(1,2);
        covar_cheetah(j,i-1) = sigmas_cheetah(2,1);
        covar_cheetah(j,j) = sigmas_cheetah(2,2);
        
        covar_grass(i-1,i-1) = sigmas_grass(1,1);
        covar_grass(i-1,j) = sigmas_grass(1,2);
        covar_grass(j,i-1) = sigmas_grass(2,1);
        covar_grass(j,j) = sigmas_grass(2,2);
    end
end

mu_cheetah_8 = [mu_cheetah(1);mu_cheetah(14);mu_cheetah(18);mu_cheetah(25);mu_cheetah(27);mu_cheetah(45);mu_cheetah(47);mu_cheetah(55)];
mu_grass_8 = [mu_grass(1);mu_grass(14);mu_grass(18);mu_grass(25);mu_grass(27);mu_grass(45);mu_grass(47);mu_grass(55)];

covar_cheetah_8 = [covar_cheetah(1,1),covar_cheetah(1,14),covar_cheetah(1,18),covar_cheetah(1,25),covar_cheetah(1,27),covar_cheetah(1,45),covar_cheetah(1,47),covar_cheetah(1,55);...
                   covar_cheetah(14,1),covar_cheetah(14,14),covar_cheetah(14,18),covar_cheetah(14,25),covar_cheetah(14,27),covar_cheetah(14,45),covar_cheetah(14,47),covar_cheetah(14,55);...
                   covar_cheetah(18,1),covar_cheetah(18,14),covar_cheetah(18,18),covar_cheetah(18,25),covar_cheetah(18,27),covar_cheetah(18,45),covar_cheetah(18,47),covar_cheetah(18,55);...
                   covar_cheetah(25,1),covar_cheetah(25,14),covar_cheetah(25,18),covar_cheetah(25,25),covar_cheetah(25,27),covar_cheetah(25,45),covar_cheetah(25,47),covar_cheetah(25,55);...
                   covar_cheetah(27,1),covar_cheetah(27,14),covar_cheetah(27,18),covar_cheetah(27,25),covar_cheetah(27,27),covar_cheetah(27,45),covar_cheetah(27,47),covar_cheetah(27,55);...
                   covar_cheetah(45,1),covar_cheetah(45,14),covar_cheetah(45,18),covar_cheetah(45,25),covar_cheetah(45,27),covar_cheetah(45,45),covar_cheetah(45,47),covar_cheetah(45,55);...
                   covar_cheetah(47,1),covar_cheetah(47,14),covar_cheetah(47,18),covar_cheetah(47,25),covar_cheetah(47,27),covar_cheetah(47,45),covar_cheetah(47,47),covar_cheetah(47,55);...
                   covar_cheetah(55,1),covar_cheetah(55,14),covar_cheetah(55,18),covar_cheetah(55,25),covar_cheetah(55,27),covar_cheetah(55,45),covar_cheetah(55,47),covar_cheetah(55,55)];

covar_grass_8 = [covar_grass(1,1),covar_grass(1,14),covar_grass(1,18),covar_grass(1,25),covar_grass(1,27),covar_grass(1,45),covar_grass(1,47),covar_grass(1,55);...
                   covar_grass(14,1),covar_grass(14,14),covar_grass(14,18),covar_grass(14,25),covar_grass(14,27),covar_grass(14,45),covar_grass(14,47),covar_grass(14,55);...
                   covar_grass(18,1),covar_grass(18,14),covar_grass(18,18),covar_grass(18,25),covar_grass(18,27),covar_grass(18,45),covar_grass(18,47),covar_grass(18,55);...
                   covar_grass(25,1),covar_grass(25,14),covar_grass(25,18),covar_grass(25,25),covar_grass(25,27),covar_grass(25,45),covar_grass(25,47),covar_grass(25,55);...
                   covar_grass(27,1),covar_grass(27,14),covar_grass(27,18),covar_grass(27,25),covar_grass(27,27),covar_grass(27,45),covar_grass(27,47),covar_grass(27,55);...
                   covar_grass(45,1),covar_grass(45,14),covar_grass(45,18),covar_grass(45,25),covar_grass(45,27),covar_grass(45,45),covar_grass(45,47),covar_grass(45,55);...
                   covar_grass(47,1),covar_grass(47,14),covar_grass(47,18),covar_grass(47,25),covar_grass(47,27),covar_grass(47,45),covar_grass(47,47),covar_grass(47,55);...
                   covar_grass(55,1),covar_grass(55,14),covar_grass(55,18),covar_grass(55,25),covar_grass(55,27),covar_grass(55,45),covar_grass(55,47),covar_grass(55,55)];

clear i j sigmas_cheetah sigmas_grass;
%% Part B_2
% Plot all the marginal distributions
margin_cheetah = cell(64,2);
margin_grass = cell(64,2);

for i = 4:4:64
    
    g = figure;
    counter = 1;
    for j = i-3:i
        
        domain_cheetah = range(TrainsampleDCT_FG(:,j));
        domain_grass = range(TrainsampleDCT_BG(:,j));
        
        pdf_cheetah = normpdf(domain_cheetah(1):0.0001:domain_cheetah(2),mu_cheetah(j),sqrt(covar_cheetah(j,j)));
        pdf_grass = normpdf(domain_grass(1):0.0001:domain_grass(2),mu_grass(j),sqrt(covar_grass(j,j)));

%     pdf_cheetah = pdf_cheetah./sum(pdf_cheetah);
%     pdf_grass = pdf_grass./sum(pdf_grass);

        subplot(2,2,counter);
        counter = counter + 1;
        plot(domain_cheetah(1):0.0001:domain_cheetah(2),pdf_cheetah,'r');
        hold on;
    
        plot(domain_grass(1):0.0001:domain_grass(2),pdf_grass,'--');
        title(['P(x_{',num2str(j),'}|(grass|cheetah))']);
%         xlabel('Domain');
%         ylabel('Probability');
%         legend(['P(x_{',num2str(j),'}|cheetah)'],['P(x_{',num2str(j),'}|grass)']);
        
        margin_cheetah{j,1} = pdf_cheetah;
        margin_cheetah{j,2} = domain_cheetah(1):0.0001:domain_cheetah(2);

        margin_grass{j,1} = pdf_grass;
        margin_grass{j,2} = domain_grass(1):0.0001:domain_grass(2);
        
    end
    
%     saveas(g,['x_',num2str(j-3),'_',num2str(j),'.jpg']);
end

clear i domain_cheetah domain_grass pdf_cheetah pdf_grass;
%% Part B_3
%Organizing the 8 best plots and the 8 worst plots

pdf_cheetah = margin_cheetah{1,1};
domain_cheetah = margin_cheetah{1,2};

pdf_grass = margin_grass{1,1};
domain_grass = margin_grass{1,2};

figure;
subplot(2,2,1);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 1');

pdf_cheetah = margin_cheetah{14,1};
domain_cheetah = margin_cheetah{14,2};

pdf_grass = margin_grass{14,1};
domain_grass = margin_grass{14,2};

subplot(2,2,2);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 14');

pdf_cheetah = margin_cheetah{18,1};
domain_cheetah = margin_cheetah{18,2};

pdf_grass = margin_grass{18,1};
domain_grass = margin_grass{18,2};

subplot(2,2,3);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 18');

pdf_cheetah = margin_cheetah{25,1};
domain_cheetah = margin_cheetah{25,2};

pdf_grass = margin_grass{25,1};
domain_grass = margin_grass{25,2};

subplot(2,2,4);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 25');

pdf_cheetah = margin_cheetah{27,1};
domain_cheetah = margin_cheetah{27,2};

pdf_grass = margin_grass{27,1};
domain_grass = margin_grass{27,2};

figure;
subplot(2,2,1);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 27');

pdf_cheetah = margin_cheetah{45,1};
domain_cheetah = margin_cheetah{45,2};

pdf_grass = margin_grass{45,1};
domain_grass = margin_grass{45,2};

subplot(2,2,2);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 45');

pdf_cheetah = margin_cheetah{47,1};
domain_cheetah = margin_cheetah{47,2};

pdf_grass = margin_grass{47,1};
domain_grass = margin_grass{47,2};

subplot(2,2,3);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 47');

pdf_cheetah = margin_cheetah{55,1};
domain_cheetah = margin_cheetah{55,2};

pdf_grass = margin_grass{55,1};
domain_grass = margin_grass{55,2};

subplot(2,2,4);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 55');

pdf_cheetah = margin_cheetah{2,1};
domain_cheetah = margin_cheetah{2,2};

pdf_grass = margin_grass{2,1};
domain_grass = margin_grass{2,2};

figure;
subplot(2,2,1);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 2');

pdf_cheetah = margin_cheetah{3,1};
domain_cheetah = margin_cheetah{3,2};

pdf_grass = margin_grass{3,1};
domain_grass = margin_grass{3,2};

subplot(2,2,2);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 3');

pdf_cheetah = margin_cheetah{4,1};
domain_cheetah = margin_cheetah{4,2};

pdf_grass = margin_grass{4,1};
domain_grass = margin_grass{4,2};

subplot(2,2,3);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 4');

pdf_cheetah = margin_cheetah{5,1};
domain_cheetah = margin_cheetah{5,2};

pdf_grass = margin_grass{5,1};
domain_grass = margin_grass{5,2};

subplot(2,2,4);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 5');

pdf_cheetah = margin_cheetah{59,1};
domain_cheetah = margin_cheetah{59,2};

pdf_grass = margin_grass{59,1};
domain_grass = margin_grass{59,2};

figure;
subplot(2,2,1);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 59');

pdf_cheetah = margin_cheetah{62,1};
domain_cheetah = margin_cheetah{62,2};

pdf_grass = margin_grass{62,1};
domain_grass = margin_grass{62,2};

subplot(2,2,2);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 62');

pdf_cheetah = margin_cheetah{63,1};
domain_cheetah = margin_cheetah{63,2};

pdf_grass = margin_grass{63,1};
domain_grass = margin_grass{63,2};

subplot(2,2,3);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 63');

pdf_cheetah = margin_cheetah{64,1};
domain_cheetah = margin_cheetah{64,2};

pdf_grass = margin_grass{64,1};
domain_grass = margin_grass{64,2};

subplot(2,2,4);
plot(domain_cheetah,pdf_cheetah,'r');
hold on;
plot(domain_grass,pdf_grass,'--');
title('Feature 64');

clear TrainsampleDCT_FG TrainsampleDCT_BG;
%% Part C_1

% Data Preparation
imagefile = 'cheetah.bmp';
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
features = zeros(S(1)*S(2),64);
vectorBlock = zeros(length(zigzag),1);
extender = 0;

for i = 1:S(1)
    for j = 1:S(2)
        block = blocks{i,j};
        block = dct2(block);
        vectorBlock(zigzag) = block;
        features(j+extender,:) = vectorBlock;
    end
    extender = extender + S(2);
end

features_8 =[features(:,1),features(:,14),features(:,18),features(:,25),features(:,27),features(:,45),features(:,47),features(:,55)];

clear ans i j image imagefile filename zigFile zigzag blocks block extender vectorBlock;
%% Part C_2

% Computing the BDR using 64-D Gaussian and 8-D Gaussian
% i*(x) = argmax_i(log(P_X|Y(x|i))+log(P_Y(i)))
L = length(features);

A = zeros(L,1);
A_8 = zeros(L,1);

denom_cheetah = log((2*pi)^(64)*det(covar_cheetah));
denom_grass = log((2*pi)^(64)*det(covar_grass));

denom_cheetah_8 = log((2*pi)^(8)*det(covar_cheetah_8));
denom_grass_8 = log((2*pi)^(8)*det(covar_grass_8));

for i = 1:L
    
    nom_cheetah = (features(i,:)'-mu_cheetah)'/covar_cheetah;
    nom_cheetah = nom_cheetah*(features(i,:)'-mu_cheetah);

    nom_grass = (features(i,:)'-mu_grass)'/covar_grass;
    nom_grass = nom_grass*(features(i,:)'-mu_grass);
    
    nom_cheetah_8 = (features_8(i,:)'-mu_cheetah_8)'/covar_cheetah_8;
    nom_cheetah_8 = nom_cheetah_8*(features_8(i,:)'-mu_cheetah_8);

    nom_grass_8 = (features_8(i,:)'-mu_grass_8)'/covar_grass_8;
    nom_grass_8 = nom_grass_8*(features_8(i,:)'-mu_grass_8);
    
    i_grass = nom_grass+denom_grass-2*log(prior_grass);
    i_cheetah = nom_cheetah+denom_cheetah-2*log(prior_cheetah);
    
    i_grass_8 = nom_grass_8+denom_grass_8-2*log(prior_grass);
    i_cheetah_8 = nom_cheetah_8+denom_cheetah_8-2*log(prior_cheetah);
    
    if(i_grass > i_cheetah)
        A(i) = 1;
    else
        A(i) = 0;
    end
    
    if(i_grass_8 > i_cheetah_8)
        A_8(i) = 1;
    else
        A_8(i) = 0;
    end
end

image_64 = zeros(S(1)+7,S(2)+7);
image_8 = image_64;
incrementer = 1;

for i = 8:S(1)+7
    for j = 8:S(2)+7
        image_64(i-7:i,j-7:j) = A(incrementer);
        image_8(i-7:i,j-7:j) = A_8(incrementer);
        incrementer = incrementer + 1;
    end
end

% image = image(1:S(1)+6,1:S(2)+6);

clear i j S L incrementer denom_cheetah denom_grass nom_cheetah nom_grass i_grass i_cheetah;
%% Part C_3
% Presenting a mask image from the computation
figure;
imagesc(image_64);
colormap(gray(255));

figure;
imagesc(image_8);
colormap(gray(255));

%% Part D_1
% Computing the error rate from the ground truth
% Error Rate = # Incorrect/# Total Pixels
imagefile = 'cheetah_mask.bmp';
figure;
truth = imread(imagefile);
imagesc(truth);
truth = im2double(truth);

S = size(truth);

detect_64 = 0;
detect_8 = 0;

error_64 = 0;
error_8 = 0;
for i = 1:S(1)
    for j = 1:S(2)
        if((truth(i,j) == image_64(i,j)) && truth(i,j) == 1)
            detect_64 = detect_64 + 1;
        end
        if((truth(i,j) == image_8(i,j)) && truth(i,j) == 1)
            detect_8 = detect_8 + 1;
        end
        if(truth(i,j) ~= image_64(i,j) && truth(i,j) == 0)
            error_64 = error_64 + 1;
        end
        if(truth(i,j) ~= image_8(i,j) && truth(i,j) == 0)
            error_8 = error_8 + 1;
        end
    end
end

size_cheetah = sum(truth);
size_cheetah = sum(size_cheetah);
p_detect_64 = detect_64/size_cheetah;
p_detect_8 = detect_8/size_cheetah;

p_error_64 = error_64/(S(1)*S(2)-size_cheetah);
p_error_8 = error_8/(S(1)*S(2)-size_cheetah);

poe_64 = (1-p_detect_64)*prior_cheetah+p_error_64*prior_grass;
poe_8 = (1-p_detect_8)*prior_cheetah+p_error_8*prior_grass;

% error_rate_64 = error_64/(S(1)*S(2));
% error_rate_8 = error_8/(S(1)*S(2));

clear imagefile i j error_64 error_8;

%% Part D_2
% Computing the probability of error of the classifier

% feat_matrix = zeros(S(1),S(2));
% incrementer = 0;

L = length(features);

for i = 1:L
   
    
end