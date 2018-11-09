%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Homework 5
% Problem 6
% ECE271A - Statistical Learning I
% Ibrahim Akbar
% 4/12/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [alphas, mus, sigmas] = initialization(C)

    alphas = zeros(C, 1);
    mus = zeros(C, 64);
    sigmas = zeros(C, 64);
    
    while(sum(alphas) < .999)
        alphas(1) = rand;

        max = 1 - alphas(1);

        for i = 2:C

            alphas(i) = rand * max;

            max = max - alphas(i);

        end

        fprintf('SUM : %f\n', sum(alphas));
    end

    for i = 1:C

        mus(i, :) = rand(1, 64)*3;
        sigmas(i, :) = rand(1, 64)*10+5;
        
    end

end
    
