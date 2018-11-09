%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Homework 5
% Problem 6
% ECE271A - Statistical Learning I
% Ibrahim Akbar
% 4/12/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data Loading

clear;
clc;
load('TrainingSamplesDCT_8_new.mat');

total = vertcat(TrainsampleDCT_FG, TrainsampleDCT_BG);

L = [length(TrainsampleDCT_FG),length(TrainsampleDCT_BG)];
prior_fg = L(1)/(L(1)+L(2));
prior_bg = L(2)/(L(1)+L(2));

%% Foreground Mixture Models

fg_mixtures = cell(6, 1);

c = [1, 2, 4, 8, 16, 32];

for j = 1:6
    
    params = cell(3,1);
    
    pf_old = 0;

    thres = 0.00001;

    [alphas_fg, mus_fg, sigmas_fg] = initialization(c(j));
    
    p_fg = zeros(1303, 1);

    for k = 1:length(alphas_fg)

        p_fg = p_fg + log(alphas_fg(k) .* mvnpdf(total, mus_fg(k, :), sigmas_fg(k, :)));

    end
    
    p_f = sum(p_fg);
    
    i = 1;
    
    while(norm(p_f - pf_old) > thres)
        
        fprintf('ITER: %d\n', i);
        i = i +1;
        
        pf_old = p_f;
        
        weights_fg = E(TrainsampleDCT_FG, alphas_fg, mus_fg, sigmas_fg);

        [alphas_fg, mus_fg, sigmas_fg] = M(TrainsampleDCT_FG, weights_fg);

        p_fg = zeros(1303, 1);

        for k = 1:length(alphas_fg)

            p_fg = p_fg + log(alphas_fg(k) .* mvnpdf(total, mus_fg(k, :), sigmas_fg(k, :)));

        end
        
        p_f = sum(p_fg);
        
    end
    
    params{1} = alphas_fg;
    params{2} = mus_fg;
    params{3} = sigmas_fg;
    
    fg_mixtures{j} = params;
    
end

clear j k i params alphas_fg mus_fg sigmas_fg weights_fg p_f pf_old p_fg;
%% Background Mixture Models

bg_mixtures = cell(6, 1);

for j = 1:6
    
    params = cell(3,1);

    [alphas_bg, mus_bg, sigmas_bg] = initialization(c(j));
    
    pf_old = 0;
    
    p_bg = zeros(1303, 1);

    for k = 1:length(alphas_bg)

        p_bg = p_bg + log(alphas_bg(k) .* mvnpdf(total, mus_bg(k, :), sigmas_bg(k, :)));

    end
    
    p_f = sum(p_bg);
    
    i = 0;

    while(norm(p_f - pf_old) > thres)
        
        fprintf('ITER: %d\n', i);
        i = i +1;
        
        pf_old = p_f;
        
        weights_bg = E(TrainsampleDCT_BG, alphas_bg, mus_bg, sigmas_bg);

        [alphas_bg, mus_bg, sigmas_bg] = M(TrainsampleDCT_BG, weights_bg);
        
        p_bg = zeros(1303, 1);

        for k = 1:length(alphas_bg)

            p_bg = p_bg + log(alphas_bg(k) .* mvnpdf(total, mus_bg(k, :), sigmas_bg(k, :)));
            
        end
        
        p_f = sum(p_bg);
    end

    params{1} = alphas_bg;
    params{2} = mus_bg;
    params{3} = sigmas_bg;

    bg_mixtures{j} = params;

end

clear i k c j thres params alphas_bg mus_bg sigmas_bg weights_bg p_f pf_old p_bg;

%% Data Preoparation

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

clear ans i j L image imagefile filename zigFile zigzag blocks block extender vectorBlock;

%% BDR Calculation 5 FG vs 1 BG

dim = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64];

L = length(features);

classA = cell(6, 1);


for l = 1:6
    
    bg_classifier = bg_mixtures{l};
    alphas_bg = bg_classifier{1};
    mus_bg = bg_classifier{2};
    sigmas_bg = bg_classifier{3};
    
    fg_classifier = fg_mixtures{l};
    alphas_fg = fg_classifier{1};
    mus_fg = fg_classifier{2};
    sigmas_fg = fg_classifier{3};

    dimA = cell(length(dim), 1);

    for i = 1:length(dim)
        
        A = zeros(L, 1);

        for j = 1:L

            p_fg = 0;
            p_bg = 0;

            for k = 1:length(alphas_fg)

                p_fg = p_fg + alphas_fg(k) * mvnpdf(features(j, 1:dim(i)), mus_fg(k, 1:dim(i)), sigmas_fg(k, 1:dim(i)));

                p_bg = p_bg + alphas_bg(k) * mvnpdf(features(j, 1:dim(i)), mus_bg(k, 1:dim(i)), sigmas_bg(k, 1:dim(i)));

            end

            i_fg = p_fg * prior_fg;

            i_bg = p_bg * prior_bg;

            if(i_bg > i_fg)
                A(j) = 0;
            else
                A(j) = 1;
            end

        end

        dimA{i} = A;
        
    end
    
    classA{l} = dimA;
    
end

clear i j p_bg p_fg i_fg i_bg mus_bg mus_fg sigmas_bg sigmas_fg alphas_bg alphas_fg ...
    fg_classifier bg_classifier dimA A k l;
%% Image Construction

imageclass = cell(length(classA), 1);

for l = 1:6
    
    dimA = classA{l};
    
    imagedim = cell(length(dimA), 1);
    
    for k = 1:length(dimA)
       
        A = dimA{k};
        image = zeros(S(1) + 7, S(2) + 7);
        incrementer = 1;

        for i = 8:S(1) + 7
            for j = 8:S(2) + 7
                image(i - 7:i, j - 7:j) = A(incrementer);
                incrementer = incrementer + 1;
            end
        end
        
        imagedim{k} = image;  
    end
    imageclass{l} = imagedim;
end


clear i j k l dimA imagedim  A incrementer;

%% Displaying Images Per Mixture
% Presenting a mask image from the computation

figure;
imagesc(image);
colormap(gray(255));

%% Probability of Error

imagefile = 'cheetah_mask.bmp';
figure;
truth = imread(imagefile);
imagesc(truth);
truth = im2double(truth);

S = size(truth);

size_fg = sum(truth);
size_fg = sum(size_fg);

PoEMix = zeros(11, 6);

for l = 1:6
    dimimage = imageclass{l};
    
    PoEdim = zeros(length(dimimage), 1);
    
    for k = 1:length(dimimage)
        
        image = dimimage{k};
        detect = 0;
        error = 0;

        for i = 1:S(1)
            for j = 1:S(2)
                if((truth(i, j) == image(i, j)) && truth(i, j) == 1)
                    detect = detect + 1;
                end
                if(truth(i, j) ~= image(i, j) && truth(i, j) == 0)
                    error = error + 1;
                end
            end
        end

        p_detect = detect / size_fg;
        p_error = error / (S(1) * S(2) - size_fg);

        poe = (1 - p_detect) * prior_fg + p_error * prior_bg;
        fprintf('PoE: %f\n',poe);
        
        PoEdim(k) = poe;
    end
    
    PoEMix(:, l) = PoEdim;
end

clear i j k l poe PoEdim truth detect error size_fg p_detect p_error;

%% Graph of PoE vs Dim

figure;
hold on;
loglog(dim, PoEMix(:, 1), '-ro');
loglog(dim, PoEMix(:, 2), '-g*');
loglog(dim, PoEMix(:, 3), '-b+');
loglog(dim, PoEMix(:, 4), '-mx');
loglog(dim, PoEMix(:, 5), '-cs');
loglog(dim, PoEMix(:, 6), '-yd');
grid on;
ax = gca;
ax.XTick = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64];
ax.XScale = 'log';
legend('C: 1', 'C: 2', 'C: 4', 'C: 8', 'C: 16', 'C: 32','Location', 'best');
xlabel('Number of Features');
ylabel('Probability of Error');
title('Probability of Error vs Components');

clear ax;