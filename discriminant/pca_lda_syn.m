%%%%%%%%%%%%%%%%%%%%%%%%%
% ECE271B - HW 1
% Ibrahim Akbar
% Winter 2018
% UCSD
%%%%%%%%%%%%%%%%%%%%%%%%%

%% Problem 4 Set Up
clear;

alpha = [10, 2];
sigma = [2, 10];

mu_1_a = [alpha(1), 0];
mu_2_a = -mu_1_a;

mu_1_b = [alpha(2), 0];
mu_2_b = -mu_1_b;

cov_1_a = [1, 0; 0, sigma(1)];
cov_2_a = cov_1_a;

cov_1_b = [1, 0; 0, sigma(2)];
cov_2_b = cov_1_b;

%% Sampling from Distributions (B)

rnd_samples_1_a = mvnrnd(mu_1_a, cov_1_a, 1000);
rnd_samples_2_a = mvnrnd(mu_2_a, cov_2_a, 1000);

rnd_samples_1_b = mvnrnd(mu_1_b, cov_1_b, 1000);
rnd_samples_2_b = mvnrnd(mu_2_b, cov_2_b, 1000);

figure(1);
hold on;
plot(rnd_samples_1_a(:,1), rnd_samples_1_a(:,2), 'mo');
plot(rnd_samples_2_a(:,1), rnd_samples_2_a(:,2), 'r+');
title('1000 Samples from Gaussian Dist. Condition: A');

figure(2);
hold on;
plot(rnd_samples_1_b(:,1), rnd_samples_1_b(:,2), 'mo');
plot(rnd_samples_2_b(:,1), rnd_samples_2_b(:,2), 'r+');
title('1000 Samples from Gaussian Dist. Condition: B');

%% PCA

X_A = [rnd_samples_1_a, rnd_samples_2_a];
X_B = [rnd_samples_1_b, rnd_samples_2_b];
S = size(X_A);

I = eye(S(2));

X_C_A = (I - (1/2)*ones(4, 4))* X_A';

X_C_B = (I - (1/2)*ones(4, 4))* X_B';


[~, E_1_a, V_1_a] = svd(X_C_A);
[~, E_1_b, V_1_b] = svd(X_C_B);

[~, i_1_a] = max(max(E_1_a));
[~, i_1_b] = max(max(E_1_b));

p_1_a = V_1_a(:, i_1_a);
p_1_b = V_1_b(:, i_1_b);

d_1_a = p_1_a' * X_A;
d_1_b = p_1_b' * X_B;

[m_a, i_a] = max(d_1_a);
[m_b, i_b] = max(d_1_b);

if(mod(i_a, 2) == 0)
    if(m_a > 0)
        x_a = 0;
        y_a = 0;
        u_a = 0;
        v_a = 5;
    else
        x_a = 0;
        y_a = 0;
        u_a = 0;
        v_a = -5;
    end
else
    if(m_a > 0)
        x_a = 0;
        y_a = 0;
        u_a = 5;
        v_a = 0;
    else
        x_a = 0;
        y_a = 0;
        u_a = -5;
        v_a = 0;
    end
end

if(mod(i_b, 2) == 0)
    if(m_b > 0)
        x_b = 0;
        y_b = 0;
        u_b = 0;
        v_b = 5;
    else
        x_b = 0;
        y_b = 0;
        u_b = 0;
        v_b = -5;
    end
else
    if(m_b > 0)
        x_b = 0;
        y_b = 0;
        u_b = 5;
        v_b = 0;
    else
        x_b = 0;
        y_b = 0;
        u_b = -5;
        v_b = 0;
    end
end

figure;
hold on;
plot(rnd_samples_1_a(:,1), rnd_samples_1_a(:,2), 'mo');
plot(rnd_samples_2_a(:,1), rnd_samples_2_a(:,2), 'r+');
quiver(x_a, y_a, u_a, v_a, 'LineWidth', 3);
title('PCA Direction Gaussian Dist. Condition: A');

figure;
hold on;
plot(rnd_samples_1_b(:,1), rnd_samples_1_b(:,2), 'mo');
plot(rnd_samples_2_b(:,1), rnd_samples_2_b(:,2), 'r+');
quiver(x_b, y_b, u_b, v_b, 'LineWidth', 3);
title('PCA Direction from Gaussian Dist. Condition: B');

%% LDA

w_a = (cov_1_a)\(mu_2_a - mu_1_a)';

w_b = (cov_1_b)\(mu_2_b - mu_1_b)';

figure;
hold on;
plot(rnd_samples_1_a(:,1), rnd_samples_1_a(:,2), 'mo');
plot(rnd_samples_2_a(:,1), rnd_samples_2_a(:,2), 'r+');
quiver(0, 0, w_a(1)/5, w_a(2), 'LineWidth', 3);
title('LDA Direction Gaussian Dist. Condition: A');

figure;
hold on;
plot(rnd_samples_1_b(:,1), rnd_samples_1_b(:,2), 'mo');
plot(rnd_samples_2_b(:,1), rnd_samples_2_b(:,2), 'r+');
quiver(0, 0, w_b(1), w_b(2), 'LineWidth', 3);
title('LDA Direction from Gaussian Dist. Condition: B');

clear i_1_a i_1_b S I U_1_a U_1_b;