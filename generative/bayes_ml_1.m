%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Homework 1
% Problem 5
% ECE271A - Statistical Learning I
% Ibrahim Akbar
% 15/10/2017
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
load('TrainingSamplesDCT_8.mat');

%IDEA 1
L = [length(TrainsampleDCT_BG), length(TrainsampleDCT_FG)];
prior_cheetah = L(2)/(L(1)+L(2));
prior_grass = L(1)/(L(1)+L(2));

%% Part B
% Finding the class conditional probabilities through computing the
% respective histograms using a 1D feature.
% max2loc() uses abs() currently, may or may not be necessary
probs_grass = zeros(L(1),1);
probs_cheetah = zeros(L(2),1);

for i = 1:length(L)
    for j = 1:L(i)
        if(i ~= 1)
            probs_cheetah(j) = max2loc(TrainsampleDCT_FG(j,:));
        else
            probs_grass(j) = max2loc(TrainsampleDCT_BG(j,:));
        end
    end
end

% range_cheetah = range(probs_cheetah);
% range_grass = range(probs_grass);

% Must adjust histograms with bin sizes (i.e. 10,15,x)
% You can specify the bin edges to reduce the error rate even more and in
% turn reduce the probability for error.
% Find the optimal size -> best resolution of distribution
% Need to scale it to adhere to a probability space (i.e. P(Omega) = 1))
% edges_cheetah = [0 3.7 7.4 11.1 14.8 18.5 22.2 25.9 29.6 33.3];
%edge_cheetah = [0 3.68888889 7.37777778 11.0666667 14.7555556 18.4444444 22.13333333 25.8222222 29.5111111 33.2];
% edges_grass = [2 3.8 5.6 7.4 9.2 11 12.8 14.6 16.4 18.2];
% Bin boundaries must be the same in order for comparison to work!
figure(1);
histogram(probs_cheetah,8,'Normalization','probability');
title('P(x|cheetah)');
ylabel('Probability of Ranges');
xlabel('Indexes of 2nd largest Coefficients');
[N_cheetah, edges_cheetah] = histcounts(probs_cheetah,8,'Normalization','probability');
% cheetah = histfit(probs_cheetah,20);
% pdf_cheetah = fitdist(probs_cheetah, 'Normal');

figure(2);
histogram(probs_grass,edges_cheetah,'Normalization','probability');
title('P(x|grass)');
ylabel('Probability of Ranges');
xlabel('Indexes of 2nd largest Coefficients');
[N_grass, edges_grass] = histcounts(probs_grass,edges_cheetah,'Normalization','probability');
% grass = histfit(probs_grass,20);
% pdf_grass = fitdist(probs_grass, 'Normal');

clear i j L range_cheetah range_grass probs_cheetah probs_grass TrainsampleDCT_FG TrainsampleDCT_BG;

%% Part C_1
% Computing the feature X 

% Data Preparation
imagefile = 'cheetah.bmp';
filename = 'Zig-Zag Pattern.txt';

zigFile = fopen(filename, 'r');
zigzag = fscanf(zigFile, '%d');
fclose(zigFile);

zigzag = zigzag + ones(length(zigzag),1);

image = imread(imagefile);
figure(3);
imagesc(image);
image = im2double(image);

% Padding may not be necessary
% image = [zeros(255,1), image, zeros(255,1)];
% image = [zeros(1,272); image; zeros(1,272)];
image = [image,zeros(255,1)];
image = [image;zeros(1,271)];

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
features = cell(S(1)*S(2),1);
vectorBlock = zeros(length(zigzag),1);
extender = 0;

for i = 1:S(1)
    for j = 1:S(2)
        block = blocks{i,j};
        block = dct2(block);
        vectorBlock(zigzag) = block;
        features{j+extender} = vectorBlock;
    end
    extender = extender + S(2);
end

% Feature X computation (2nd largest magnitude of coefficients)
L = length(features);
feature = zeros(L,1);

for i = 1:L
    vector = features{i};
    feature(i) = max2loc(vector);
end

clear i j features image imagefile S vector vectorBlock zigFile zigzag ans block extender filename;

%% Part C_2
% Computing the state variable Y
% Storing values into a vector A
% i*(x) = argmax_i(log(P_X|Y(x|i))+log(P_Y(i)))
A = zeros(L,1);

% mu_cheetah = pdf_cheetah.mu;
% mu_grass = pdf_grass.mu;
% sigma_cheetah = pdf_cheetah.sigma;
% sigma_grass = pdf_grass.sigma;
%  
% comp_1_cheetah = (1/2)*log(2*pi*sigma_cheetah^2);
% comp_1_grass = (1/2)*log(2*pi*sigma_grass^2);

for i = 1:length(feature)
    prob_cheetah = 0;
    prob_grass = 0;
    
    for j = 2:length(edges_cheetah)
        if(feature(i) < edges_cheetah(j))
            prob_cheetah = N_cheetah(j-1);
            break;
        end
    end
    
    for j = 2:length(edges_grass)
        if(feature(i) < edges_grass(j))
            prob_grass = N_grass(j-1);
            break;
        end
    end
        
    i_cheetah = log(prob_cheetah)+log(prior_cheetah);
    i_grass = log(prob_grass)+log(prior_grass);
        
%      comp_2_cheetah = ((feature(i)-mu_cheetah)/sigma_cheetah)^2;
%      i_cheetah = comp_1_cheetah+comp_2_cheetah-log(prior_cheetah);
%      
%      comp_2_grass = ((feature(i)-mu_grass)/sigma_grass)^2;
%     i_grass = comp_1_grass+comp_2_grass-log(prior_grass);
    
    if(i_grass > i_cheetah)
        A(i) = 0;
    else
        A(i) = 1;
    end
end

S = size(blocks);
image = zeros(S(1)+7,S(2)+7);
incrementer = 1;

for i = 8:S(1)+7
    for j = 8:S(2)+7
        image(i-7:i,j-7:j) = A(incrementer);
        incrementer = incrementer + 1;
    end
end

image = image(1:S(1)+6,1:S(2)+6);

clear i i_cheetah i_grass comp_1_cheetah comp_1_grass comp_2_cheetah comp_2_grass;

%% Part C_3
% Presenting a mask image from the computation
figure(6);
imagesc(image);
colormap(gray(255));

%% Part D_1
% Computing the error rate from the ground truth
% Error Rate = # Incorrect/# Total Pixels
imagefile = 'cheetah_mask.bmp';
figure(5);
truth = imread(imagefile);
imagesc(truth);
truth = im2double(truth);

S = size(truth);

error = 0;
for i = 1:S(1)
    for j = 1:S(2)
        if(truth(i,j) ~= image(i,j))
            error = error + 1;
        end
    end
end

error_rate = error/(S(1)*S(2));

%% Part D_2
% Computing the probability of error from the ground truthl...
% Probability of error = Integral(Space: R2 p(x|w1)p(w1)dx)+Integral(Space:
feat_matrix = zeros(S(1)+1,S(2)+1);
incrementer = 1;

for i = 8:S(1)+1
    for j = 8:S(2)+1
        feat_matrix(i-7:i,j-7:j) = feature(incrementer);
        incrementer = incrementer + 1;
    end
end

feat_matrix = feat_matrix(1:S(1),1:S(2));
counter = 0;
gfeats_checked = zeros(33,1);
cfeats_checked = zeros(33,1);
gchecked = 1;
cchecked = 1;
false_grass = zeros(S(1)*S(2),1);
false_cheetah = zeros(S(1)*S(2),1);
for i = 1:S(1)
    for j = 1:S(2)
        if(truth(i,j) ~= image(i,j))
            if(truth(i,j) == 1)
                for k = 2:length(edges_cheetah)
                    if(feat_matrix(i,j) < edges_cheetah(k))
                        if(isempty(find(cfeats_checked == feat_matrix(i,j),1)))
                            cfeats_checked(cchecked) = feat_matrix(i,j);
                            cchecked = cchecked + 1;
                            false_grass(j+counter) = N_cheetah(k-1)*prior_cheetah;
                            break;
                        end
                    end
                end
            elseif(truth(i,j) == 0)
                for k = 2:length(edges_grass)
                    if(feat_matrix(i,j) < edges_grass(k))
                        if(isempty(find(gfeats_checked == feat_matrix(i,j),1)))
                            gfeats_checked(gchecked) = feat_matrix(i,j);
                            gchecked = gchecked + 1;
                            false_cheetah(j+counter) = N_grass(k-1)*prior_grass;
                            break;
                        end
                    end 
                end
            end
        end
    end
    counter = counter + S(2);
end

false_cheetah(false_cheetah == 0) = [];
false_grass(false_grass == 0) = [];
prob_error = sum(false_grass)+sum(false_cheetah);

