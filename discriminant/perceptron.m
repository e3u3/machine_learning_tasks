%%%%%%%%%%%%%%%%%%%%%%
% ECE271B - ML II
% HW 2
% Ibrahim Akbar
% Winter 2018
% UCSD
%%%%%%%%%%%%%%%%%%%%%%

%% Loading Data

clear;

ImageFile = gunzip('.\training_set\train-images-idx3-ubyte.gz');
LabelFile = gunzip('.\training_set\train-labels-idx1-ubyte.gz');
[train_images, train_labels] = readMNIST(ImageFile{1}, LabelFile{1}, 60000, 0);

ImageFile = gunzip('.\test_set\t10k-images-idx3-ubyte.gz');
LabelFile = gunzip('.\test_set\t10k-labels-idx1-ubyte.gz');
[test_images, test_labels] = readMNIST(ImageFile{1}, LabelFile{1}, 10000, 0);

% clear ImageFile LabelFile;

%% 1 Layer Network

t_space_size = size(train_images);

w = randn(t_space_size(2),10);
grad = zeros(10,t_space_size(2));
a = zeros(10,1);

test_errors = [];
train_errors = [];

t = 0;

labels = eye(10);

eta = 10^(-5);

flag = 0;

while(flag == 0)
    
    test_error = 0;
    train_error = 0;
    
    for i = 1:t_space_size(1)
        
        if(i <= length(test_labels))
            test_l = test_labels(i);
            a = w'*test_images(i,:)';
            y = softmax(a);
            [~,i_n] = max(y);
            if(i_n-1 ~= test_l)
                test_error = test_error + 1;
            end
        end
            
        train_l = train_labels(i);
        
        c_label = labels(:,train_l+1);
        
        a_t = w'*train_images(i,:)';
        
        y_t = softmax(a_t);
        
        [~,i_t] = max(y_t);
        
        if(i_t-1 ~= train_l)
            train_error = train_error + 1;
        end
    
        for j = 1:10
            
            grad(j,:) = grad(j,:) + (c_label(j)-y_t(j)).*train_images(i,:);
            
        end
    
    end
    
    test_errors = [test_errors; test_error/length(test_labels)];
    train_errors = [train_errors; train_error/t_space_size(1)];
    
    len_error = length(train_errors);
    if (len_error > 1)
        error = train_errors(len_error);
    else
        error = 10;
    end
    
    disp(['Error: ',num2str(train_errors(len_error)),' Time: ',num2str(t)]);
    if(error < 0.01)
        flag = 1;
    else
        w = w + eta.*grad';
    end

    grad = zeros(10,t_space_size(2));
    t = t + 1;
    
end

%% 2 Layer Network

t_space_size = size(train_images);

% w_1_1 = randn(t_space_size(2),10);
w_1_2 = randn(t_space_size(2),20);
% w_1_5 = randn(t_space_size(2),50);

% w_2_1 = randn(10,10);
w_2_2 = randn(20,10);
% w_2_5 = randn(50,10);

% grad_2_1 = zeros(10,10);
grad_2_2 = zeros(10,20);
% grad_2_5 = zeros(10,50);

% grad_1_1 = zeros(10,t_space_size(2));
grad_1_2 = zeros(20,t_space_size(2));
% grad_1_5 = zeros(50,t_space_size(2));

% test_errors_1 = [];
test_errors_2 = [];
% test_errors_5 = [];
% train_errors_1 = [];
train_errors_2 = [];
% train_errors_5 = [];

t = 0;

labels = eye(10);

eta = 10^(-5);

flag = 0;

while(flag == 0)
    
%     test_error_1 = 0;
    test_error_2 = 0;
%     test_error_5 = 0;

%     train_error_1 = 0;
    train_error_2 = 0;
%     train_error_5 = 0;
    
    for i = 1:t_space_size(1)
        
        if(i <= length(test_labels))
            
            test_l = test_labels(i);
%             h_units_1_t = w_1_1'*test_images(i,:)';
            h_units_2_t = w_1_2'*test_images(i,:)';
%             h_units_5_t = w_1_5'*test_images(i,:)';
            
%             r_units_1_t = sigmf(h_units_1_t,[1 0]);
            r_units_2_t = sigmf(h_units_2_t,[1 0]);
%             r_units_5_t = sigmf(h_units_5_t,[1 0]);
            
%             v_1_t = w_2_1'*r_units_1_t;
            v_2_t = w_2_2'*r_units_2_t;
%             v_5_t = w_2_5'*r_units_5_t;
            
%             y_1_t = softmax(v_1_t);
            y_2_t = softmax(v_2_t);
%             y_5_t = softmax(v_5_t);
            
%             [~,i_n_1] = max(y_1_t);
            [~,i_n_2] = max(y_2_t);
%             [~,i_n_5] = max(y_5_t);
            
%             if(i_n_1-1 ~= test_l)
%                 test_error_1 = test_error_1 + 1;
%             end
            if(i_n_2-1 ~= test_l)
                test_error_2 = test_error_2 + 1;
            end
%             if(i_n_5-1 ~= test_l)
%                 test_error_5 = test_error_5 + 1;
%             end
        end
         
        train_l = train_labels(i);
        
        c_label = labels(:,train_l+1);
        
%         h_units_1 = w_1_1'*train_images(i,:)';
        h_units_2 = w_1_2'*train_images(i,:)';
%         h_units_5 = w_1_5'*train_images(i,:)';
        
%         r_units_1 = sigmf(h_units_1,[1 0]);
        r_units_2 = sigmf(h_units_2,[1 0]);
%         r_units_5 = sigmf(h_units_5,[1 0]);
        
%         v_1 = w_2_1'*r_units_1;
        v_2 = w_2_2'*r_units_2;
%         v_5 = w_2_5'*r_units_5;

%         y_1 = softmax(v_1);
        y_2 = softmax(v_2);
%         y_5 = softmax(v_5);

%         [~,i_t_1] = max(y_1);
        [~,i_t_2] = max(y_2);
%         [~,i_t_5] = max(y_5);
        
%         if(i_t_1-1 ~= train_l)
%             train_error_1 = train_error_1 + 1;
%         end
        if(i_t_2-1 ~= train_l)
            train_error_2 = train_error_2 + 1;
        end
%         if(i_t_5-1 ~= train_l)
%             train_error_5 = train_error_5 + 1;
%         end  
        
%         grad_2_1 = grad_2_1 + (c_label-y_1)*r_units_1';
        grad_2_2 = grad_2_2 + (c_label-y_2)*r_units_2';
%         grad_2_5 = grad_2_5 + (c_label-y_5)*r_units_5';
        
%         grad_1_1 = grad_1_1 + ((r_units_1.*(ones(10,1)-r_units_1)).*(w_2_1*(c_label-y_1)))*train_images(i,:);
        grad_1_2 = grad_1_2 + ((r_units_2.*(ones(20,1)-r_units_2)).*(w_2_2*(c_label-y_2)))*train_images(i,:);
%         grad_1_5 = grad_1_5 + ((r_units_5.*(ones(50,1)-r_units_5)).*(w_2_5*(c_label-y_5)))*train_images(i,:);
        
    end
    
%     error_1 = train_error_1/t_space_size(1);
    error_2 = train_error_2/t_space_size(1);
%     error_3 = train_error_5/t_space_size(1);
    
%     test_errors_1 = [test_errors_1; test_error_1/length(test_labels)];
    test_errors_2 = [test_errors_2; test_error_2/length(test_labels)];
%     test_errors_5 = [test_errors_5; test_error_5/length(test_labels)];
%     train_errors_1 = [train_errors_1; error_1];
    train_errors_2 = [train_errors_2; error_2];
%     train_errors_5 = [train_errors_5; error_3];
    
%     disp(['Error 1: ',num2str(error_1),' Error 2: ',num2str(error_2),' Error 3: ',num2str(error_3),' Time: ',num2str(t)]);
    disp(['Error : ',num2str(error_2),' Time: ',num2str(t)]);
    if(error_2 < 0.08)
        flag = 1;
    else
%         w_2_1 = w_2_1 + eta.*grad_2_1';
        w_2_2 = w_2_2 + eta.*grad_2_2';
%         w_2_5 = w_2_5 + eta.*grad_2_5';
        
%         w_1_1 = w_1_1 + eta.*grad_1_1';
        w_1_2 = w_1_2 + eta.*grad_1_2';
%         w_1_5 = w_1_5 + eta.*grad_1_5';
    end

%     grad_2_1 = zeros(10,10);
    grad_2_2 = zeros(10,20);
%     grad_2_5 = zeros(10,50);

%     grad_1_1 = zeros(10,t_space_size(2));
    grad_1_2 = zeros(20,t_space_size(2));
%     grad_1_5 = zeros(50,t_space_size(2));
    t = t + 1;
    
end


%% Sigmoid Regularization

t_space_size = size(train_images);

% w_1_1 = randn(t_space_size(2),10);
% w_1_2 = randn(t_space_size(2),20);
w_1_5 = randn(t_space_size(2),50);

% w_2_1 = randn(10,10);
% w_2_2 = randn(20,10);
w_2_5 = randn(50,10);

% grad_2_1 = zeros(10,10);
% grad_2_2 = zeros(10,20);
grad_2_5 = zeros(10,50);

% grad_1_1 = zeros(10,t_space_size(2));
% grad_1_2 = zeros(20,t_space_size(2));
grad_1_5 = zeros(50,t_space_size(2));

% test_errors_1 = [];
% test_errors_2 = [];
test_errors_5 = [];
% train_errors_1 = [];
% train_errors_2 = [];
train_errors_5 = [];

t = 0;

labels = eye(10);

eta = 2*10^(-6);
lambda = 0.0001;

flag = 0;

while(flag == 0)
    
%     test_error_1 = 0;
%     test_error_2 = 0;
    test_error_5 = 0;

%     train_error_1 = 0;
%     train_error_2 = 0;
    train_error_5 = 0;
    
    for i = 1:t_space_size(1)
        
        if(i <= length(test_labels))
            
            test_l = test_labels(i);
%             h_units_1_t = w_1_1'*test_images(i,:)';
%             h_units_2_t = w_1_2'*test_images(i,:)';
            h_units_5_t = w_1_5'*test_images(i,:)';
            
%             r_units_1_t = sigmf(h_units_1_t,[1 0]);
%             r_units_2_t = sigmf(h_units_2_t,[1 0]);
            r_units_5_t = sigmf(h_units_5_t,[1 0]);
            
%             v_1_t = w_2_1'*r_units_1_t;
%             v_2_t = w_2_2'*r_units_2_t;
            v_5_t = w_2_5'*r_units_5_t;
            
%             y_1_t = softmax(v_1_t);
%             y_2_t = softmax(v_2_t);
            y_5_t = softmax(v_5_t);
            
%             [~,i_n_1] = max(y_1_t);
%             [~,i_n_2] = max(y_2_t);
            [~,i_n_5] = max(y_5_t);
            
%             if(i_n_1-1 ~= test_l)
%                 test_error_1 = test_error_1 + 1;
%             end
%             if(i_n_2-1 ~= test_l)
%                 test_error_2 = test_error_2 + 1;
%             end
            if(i_n_5-1 ~= test_l)
                test_error_5 = test_error_5 + 1;
            end
        end
         
        train_l = train_labels(i);
        
        c_label = labels(:,train_l+1);
        
%         h_units_1 = w_1_1'*train_images(i,:)';
%         h_units_2 = w_1_2'*train_images(i,:)';
        h_units_5 = w_1_5'*train_images(i,:)';
        
%         r_units_1 = sigmf(h_units_1,[1 0]);
%         r_units_2 = sigmf(h_units_2,[1 0]);
        r_units_5 = sigmf(h_units_5,[1 0]);
        
%         v_1 = w_2_1'*r_units_1;
%         v_2 = w_2_2'*r_units_2;
        v_5 = w_2_5'*r_units_5;

%         y_1 = softmax(v_1);
%         y_2 = softmax(v_2);
        y_5 = softmax(v_5);

%         [~,i_t_1] = max(y_1);
%         [~,i_t_2] = max(y_2);
        [~,i_t_5] = max(y_5);
        
%         if(i_t_1-1 ~= train_l)
%             train_error_1 = train_error_1 + 1;
%         end
%         if(i_t_2-1 ~= train_l)
%             train_error_2 = train_error_2 + 1;
%         end
        if(i_t_5-1 ~= train_l)
            train_error_5 = train_error_5 + 1;
        end  
        
%         grad_2_1 = grad_2_1 + (c_label-y_1)*r_units_1';
%         grad_2_2 = grad_2_2 + (c_label-y_2)*r_units_2';
        grad_2_5 = grad_2_5 + (c_label-y_5)*r_units_5';
        
%         grad_1_1 = grad_1_1 + ((r_units_1.*(ones(10,1)-r_units_1)).*(w_2_1*(c_label-y_1)))*train_images(i,:);
%         grad_1_2 = grad_1_2 + ((r_units_2.*(ones(20,1)-r_units_2)).*(w_2_2*(c_label-y_2)))*train_images(i,:);
        grad_1_5 = grad_1_5 + ((r_units_5.*(ones(50,1)-r_units_5)).*(w_2_5*(c_label-y_5)))*train_images(i,:);
        
    end
    
%     error_1 = train_error_1/t_space_size(1);
%     error_2 = train_error_2/t_space_size(1);
    error_3 = train_error_5/t_space_size(1);
    
%     test_errors_1 = [test_errors_1; test_error_1/length(test_labels)];
%     test_errors_2 = [test_errors_2; test_error_2/length(test_labels)];
    test_errors_5 = [test_errors_5; test_error_5/length(test_labels)];
%     train_errors_1 = [train_errors_1; error_1];
%     train_errors_2 = [train_errors_2; error_2];
    train_errors_5 = [train_errors_5; error_3];
    
%     disp(['Error 1: ',num2str(error_1),' Error 2: ',num2str(error_2),' Error 3: ',num2str(error_3),' Time: ',num2str(t)]);
    disp(['Error : ',num2str(error_3),' Time: ',num2str(t)]);
    if(error_3 < 0.1)
        flag = 1;
    else
%         w_2_1 = w_2_1 + eta.*grad_2_1' - eta*lambda*w_2_1;
%         w_2_2 = w_2_2 + eta.*grad_2_2' - eta*lambda*w_2_2;
        w_2_5 = w_2_5 + eta.*grad_2_5' - eta*lambda*w_2_5;
        
%         w_1_1 = w_1_1 + eta.*grad_1_1' - eta*lambda*w_1_1;
%         w_1_2 = w_1_2 + eta.*grad_1_2' - eta*lambda*w_1_2;
        w_1_5 = w_1_5 + eta.*grad_1_5' - eta*lambda*w_1_5;
    end

%     grad_2_1 = zeros(10,10);
%     grad_2_2 = zeros(10,20);
    grad_2_5 = zeros(10,50);

%     grad_1_1 = zeros(10,t_space_size(2));
%     grad_1_2 = zeros(20,t_space_size(2));
    grad_1_5 = zeros(50,t_space_size(2));
    t = t + 1;
    
end

%% ReLU Regularization

t_space_size = size(train_images);

w_1_1 = randn(t_space_size(2),10);
% w_1_2 = randn(t_space_size(2),20);
% w_1_5 = randn(t_space_size(2),50);

w_2_1 = randn(10,10);
% w_2_2 = randn(20,10);
% w_2_5 = randn(50,10);

grad_2_1 = zeros(10,10);
% grad_2_2 = zeros(10,20);
% grad_2_5 = zeros(10,50);

grad_1_1 = zeros(10,t_space_size(2));
% grad_1_2 = zeros(20,t_space_size(2));
% grad_1_5 = zeros(50,t_space_size(2));

test_errors_1 = [];
% test_errors_2 = [];
% test_errors_5 = [];
train_errors_1 = [];
% train_errors_2 = [];
% train_errors_5 = [];

t = 0;

labels = eye(10);

eta = 2*10^(-6);
lambda = 0.001;

flag = 0;

while(flag == 0)
    
    test_error_1 = 0;
%     test_error_2 = 0;
%     test_error_5 = 0;

    train_error_1 = 0;
%     train_error_2 = 0;
%     train_error_5 = 0;
    
    for i = 1:t_space_size(1)
        
        if(i <= length(test_labels))
            
            test_l = test_labels(i);
            h_units_1_t = w_1_1'*test_images(i,:)';
%             h_units_2_t = w_1_2'*test_images(i,:)';
%             h_units_5_t = w_1_5'*test_images(i,:)';
            
            r_units_1_t = h_units_1_t;
%             r_units_2_t = h_units_2_t;
%             r_units_5_t = h_units_5_t;
            
            zero_t_1 = find(h_units_1_t <= 0);
%             zero_t_2 = find(h_units_2_t <= 0);
%             zero_t_5 = find(h_units_5_t <= 0);
            
            for j = 1:length(zero_t_1)
                r_units_1_t(zero_t_1(j)) = 0.01;
            end
%             for j = 1:length(zero_t_2)
%                 r_units_2_t(zero_t_2(j)) = 0;
%             end
%             for j = 1:length(zero_t_5)
%                 r_units_5_t(zero_t_5(j)) = 0;
%             end
            
            v_1_t = w_2_1'*r_units_1_t;
%             v_2_t = w_2_2'*r_units_2_t;
%             v_5_t = w_2_5'*r_units_5_t;
            
            y_1_t = softmax(v_1_t);
%             y_2_t = softmax(v_2_t);
%             y_5_t = softmax(v_5_t);
            
            [~,i_n_1] = max(y_1_t);
%             [~,i_n_2] = max(y_2_t);
%             [~,i_n_5] = max(y_5_t);
            
            if(i_n_1-1 ~= test_l)
                test_error_1 = test_error_1 + 1;
            end
%             if(i_n_2-1 ~= test_l)
%                 test_error_2 = test_error_2 + 1;
%             end
%             if(i_n_5-1 ~= test_l)
%                 test_error_5 = test_error_5 + 1;
%             end
        end
         
        train_l = train_labels(i);
        
        c_label = labels(:,train_l+1);
        
        h_units_1 = w_1_1'*train_images(i,:)';
%         h_units_2 = w_1_2'*train_images(i,:)';
%         h_units_5 = w_1_5'*train_images(i,:)';
        
        r_units_1 = h_units_1;
%         r_units_2 = h_units_2;
%         r_units_5 = h_units_5;
        
        zero_1 = find(h_units_1 <= 0);
%         zero_2 = find(h_units_2 <= 0);
%         zero_5 = find(h_units_5 <= 0);
        for j = 1:length(zero_1)
            r_units_1(zero_1(j)) = 0.01;
        end
%         for j = 1:length(zero_2)
%             r_units_2(zero_2(j)) = 0;
%         end
%         for j = 1:length(zero_t)
%             r_units_5(zero_t(j)) = 0;
%         end
        
        v_1 = w_2_1'*r_units_1;
%         v_2 = w_2_2'*r_units_2;
%         v_5 = w_2_5'*r_units_5;

        y_1 = softmax(v_1);
%         y_2 = softmax(v_2);
%         y_5 = softmax(v_5);

        [~,i_t_1] = max(y_1);
%         [~,i_t_2] = max(y_2);
%         [~,i_t_5] = max(y_5);
        
        if(i_t_1-1 ~= train_l)
            train_error_1 = train_error_1 + 1;
        end
%         if(i_t_2-1 ~= train_l)
%             train_error_2 = train_error_2 + 1;
%         end
%         if(i_t_5-1 ~= train_l)
%             train_error_5 = train_error_5 + 1;
%         end  
        
        grad_2_1 = grad_2_1 + (c_label-y_1)*r_units_1';
%         grad_2_2 = grad_2_2 + (c_label-y_2)*r_units_2';
%         grad_2_5 = grad_2_5 + (c_label-y_5)*r_units_5';
        
        grad_1_1 = grad_1_1 + ((r_units_1.*(ones(10,1)-r_units_1)).*(w_2_1*(c_label-y_1)))*train_images(i,:);
%         grad_1_2 = grad_1_2 + ((r_units_2.*(ones(20,1)-r_units_2)).*(w_2_2*(c_label-y_2)))*train_images(i,:);
%         grad_1_5 = grad_1_5 + ((r_units_5.*(ones(50,1)-r_units_5)).*(w_2_5*(c_label-y_5)))*train_images(i,:);
%         
    end
    
    error_1 = train_error_1/t_space_size(1);
%     error_2 = train_error_2/t_space_size(1);
%     error_3 = train_error_5/t_space_size(1);
    
    test_errors_1 = [test_errors_1; test_error_1/length(test_labels)];
%     test_errors_2 = [test_errors_2; test_error_2/length(test_labels)];
%     test_errors_5 = [test_errors_5; test_error_5/length(test_labels)];
    train_errors_1 = [train_errors_1; error_1];
%     train_errors_2 = [train_errors_2; error_2];
%     train_errors_5 = [train_errors_5; error_3];
    
%     disp(['Error 1: ',num2str(error_1),' Error 2: ',num2str(error_2),' Error 3: ',num2str(error_3),' Time: ',num2str(t)]);
    disp(['Error : ',num2str(error_1),' Time: ',num2str(t)]);
    if(error_1 < 0.1)
        flag = 1;
    else
        w_2_1 = w_2_1 + eta.*grad_2_1' - lambda*w_2_1;
%         w_2_2 = w_2_2 + eta.*grad_2_2' - eta*lamba*w_2_2;
%         w_2_5 = w_2_5 + eta.*grad_2_5' - eta*lamba*w_2_5;
        
        w_1_1 = w_1_1 + eta.*grad_1_1' - lambda*w_1_1;
%         w_1_2 = w_1_2 + eta.*grad_1_2' - eta*lamba*w_1_2;
%         w_1_5 = w_1_5 + eta.*grad_1_5' - eta*lamba*w_1_5;
    end

    grad_2_1 = zeros(10,10);
%     grad_2_2 = zeros(10,20);
%     grad_2_5 = zeros(10,50);

    grad_1_1 = zeros(10,t_space_size(2));
%     grad_1_2 = zeros(20,t_space_size(2));
%     grad_1_5 = zeros(50,t_space_size(2));
    t = t + 1;
    
end

%% Sigmoid Regularization even Moooreeeee

t_space_size = size(train_images);

w_1_1 = randn(t_space_size(2),10);
w_1_2 = randn(t_space_size(2),20);
w_1_5 = randn(t_space_size(2),50);

w_2_1 = randn(10,10);
w_2_2 = randn(20,10);
w_2_5 = randn(50,10);

grad_2_1 = zeros(10,10);
grad_2_2 = zeros(10,20);
grad_2_5 = zeros(10,50);

grad_1_1 = zeros(10,t_space_size(2));
grad_1_2 = zeros(20,t_space_size(2));
grad_1_5 = zeros(50,t_space_size(2));

test_errors_1 = [];
test_errors_2 = [];
test_errors_5 = [];
train_errors_1 = [];
train_errors_2 = [];
train_errors_5 = [];

t = 0;

labels = eye(10);

eta = 2*10^(-6);
lambda = 0.0001;

flag = 0;

while(flag == 0)
    
    test_error_1 = 0;
    test_error_2 = 0;
    test_error_5 = 0;

    train_error_1 = 0;
    train_error_2 = 0;
    train_error_5 = 0;
    
    for i = 1:t_space_size(1)
        
        if(i <= length(test_labels))
            
            test_l = test_labels(i);
            h_units_1_t = w_1_1'*test_images(i,:)';
            h_units_2_t = w_1_2'*test_images(i,:)';
            h_units_5_t = w_1_5'*test_images(i,:)';
            
            r_units_1_t = sigmf(h_units_1_t,[1 0]);
            r_units_2_t = sigmf(h_units_2_t,[1 0]);
            r_units_5_t = sigmf(h_units_5_t,[1 0]);
            
            v_1_t = w_2_1'*r_units_1_t;
            v_2_t = w_2_2'*r_units_2_t;
            v_5_t = w_2_5'*r_units_5_t;
            
            y_1_t = softmax(v_1_t);
            y_2_t = softmax(v_2_t);
            y_5_t = softmax(v_5_t);
            
            [~,i_n_1] = max(y_1_t);
            [~,i_n_2] = max(y_2_t);
            [~,i_n_5] = max(y_5_t);
            
            if(i_n_1-1 ~= test_l)
                test_error_1 = test_error_1 + 1;
            end
            if(i_n_2-1 ~= test_l)
                test_error_2 = test_error_2 + 1;
            end
            if(i_n_5-1 ~= test_l)
                test_error_5 = test_error_5 + 1;
            end
        end
         
        train_l = train_labels(i);
        
        c_label = labels(:,train_l+1);
        
        h_units_1 = w_1_1'*train_images(i,:)';
        h_units_2 = w_1_2'*train_images(i,:)';
        h_units_5 = w_1_5'*train_images(i,:)';
        
        r_units_1 = sigmf(h_units_1,[1 0]);
        r_units_2 = sigmf(h_units_2,[1 0]);
        r_units_5 = sigmf(h_units_5,[1 0]);
        
        v_1 = w_2_1'*r_units_1;
        v_2 = w_2_2'*r_units_2;
        v_5 = w_2_5'*r_units_5;

        y_1 = softmax(v_1);
        y_2 = softmax(v_2);
        y_5 = softmax(v_5);

        [~,i_t_1] = max(y_1);
        [~,i_t_2] = max(y_2);
        [~,i_t_5] = max(y_5);
        
        if(i_t_1-1 ~= train_l)
            train_error_1 = train_error_1 + 1;
        end
        if(i_t_2-1 ~= train_l)
            train_error_2 = train_error_2 + 1;
        end
        if(i_t_5-1 ~= train_l)
            train_error_5 = train_error_5 + 1;
        end  
        
        grad_2_1 = grad_2_1 + (c_label-y_1)*r_units_1';
        grad_2_2 = grad_2_2 + (c_label-y_2)*r_units_2';
        grad_2_5 = grad_2_5 + (c_label-y_5)*r_units_5';
        
        grad_1_1 = grad_1_1 + ((r_units_1.*(ones(10,1)-r_units_1)).*(w_2_1*(c_label-y_1)))*train_images(i,:);
        grad_1_2 = grad_1_2 + ((r_units_2.*(ones(20,1)-r_units_2)).*(w_2_2*(c_label-y_2)))*train_images(i,:);
        grad_1_5 = grad_1_5 + ((r_units_5.*(ones(50,1)-r_units_5)).*(w_2_5*(c_label-y_5)))*train_images(i,:);
        
    end
    
    error_1 = train_error_1/t_space_size(1);
    error_2 = train_error_2/t_space_size(1);
    error_3 = train_error_5/t_space_size(1);
    
    test_errors_1 = [test_errors_1; test_error_1/length(test_labels)];
    test_errors_2 = [test_errors_2; test_error_2/length(test_labels)];
    test_errors_5 = [test_errors_5; test_error_5/length(test_labels)];
    train_errors_1 = [train_errors_1; error_1];
    train_errors_2 = [train_errors_2; error_2];
    train_errors_5 = [train_errors_5; error_3];
    
    disp(['Error 1: ',num2str(error_1),' Error 2: ',num2str(error_2),' Error 3: ',num2str(error_3),' Time: ',num2str(t)]);
%     disp(['Error : ',num2str(error_2),' Time: ',num2str(t)]);
    if(error_1 < 0.1)
        flag = 1;
    else
        w_2_1 = w_2_1 + eta.*grad_2_1' - eta*lamba*w_2_1;
        w_2_2 = w_2_2 + eta.*grad_2_2' - eta*lamba*w_2_2;
        w_2_5 = w_2_5 + eta.*grad_2_5' - eta*lamba*w_2_5;
        
        w_1_1 = w_1_1 + eta.*grad_1_1' - eta*lamba*w_1_1;
        w_1_2 = w_1_2 + eta.*grad_1_2' - eta*lamba*w_1_2;
        w_1_5 = w_1_5 + eta.*grad_1_5' - eta*lamba*w_1_5;
    end

    grad_2_1 = zeros(10,10);
    grad_2_2 = zeros(10,20);
    grad_2_5 = zeros(10,50);

    grad_1_1 = zeros(10,t_space_size(2));
    grad_1_2 = zeros(20,t_space_size(2));
    grad_1_5 = zeros(50,t_space_size(2));
    t = t + 1;
    
end

%% ReLU Regularization Even Moooorrreeee

t_space_size = size(train_images);

w_1_1 = randn(t_space_size(2),10);
w_1_2 = randn(t_space_size(2),20);
w_1_5 = randn(t_space_size(2),50);

w_2_1 = randn(10,10);
w_2_2 = randn(20,10);
w_2_5 = randn(50,10);

grad_2_1 = zeros(10,10);
grad_2_2 = zeros(10,20);
grad_2_5 = zeros(10,50);

grad_1_1 = zeros(10,t_space_size(2));
grad_1_2 = zeros(20,t_space_size(2));
grad_1_5 = zeros(50,t_space_size(2));

test_errors_1 = [];
test_errors_2 = [];
test_errors_5 = [];
train_errors_1 = [];
train_errors_2 = [];
train_errors_5 = [];

t = 0;

labels = eye(10);

eta = 2*10^(-6);
lambda = 0.0001;

flag = 0;

while(flag == 0)
    
    test_error_1 = 0;
    test_error_2 = 0;
    test_error_5 = 0;

    train_error_1 = 0;
    train_error_2 = 0;
    train_error_5 = 0;
    
    for i = 1:t_space_size(1)
        
        if(i <= length(test_labels))
            
            test_l = test_labels(i);
            h_units_1_t = w_1_1'*test_images(i,:)';
            h_units_2_t = w_1_2'*test_images(i,:)';
            h_units_5_t = w_1_5'*test_images(i,:)';
            
            r_units_1_t = h_units_1_t;
            r_units_2_t = h_units_2_t;
            r_units_5_t = h_units_5_t;
            
            zero_t_1 = find(h_units_1_t <= 0);
            zero_t_2 = find(h_units_2_t <= 0);
            zero_t_5 = find(h_units_5_t <= 0);
            
            for j = 1:length(zero_t_1)
                r_units_1_t(zero_t_1(j)) = 0;
            end
            for j = 1:length(zero_t_2)
                r_units_2_t(zero_t_2(j)) = 0;
            end
            for j = 1:length(zero_t_5)
                r_units_5_t(zero_t_5(j)) = 0;
            end
            
            v_1_t = w_2_1'*r_units_1_t;
            v_2_t = w_2_2'*r_units_2_t;
            v_5_t = w_2_5'*r_units_5_t;
            
            y_1_t = softmax(v_1_t);
            y_2_t = softmax(v_2_t);
            y_5_t = softmax(v_5_t);
            
            [~,i_n_1] = max(y_1_t);
            [~,i_n_2] = max(y_2_t);
            [~,i_n_5] = max(y_5_t);
            
            if(i_n_1-1 ~= test_l)
                test_error_1 = test_error_1 + 1;
            end
            if(i_n_2-1 ~= test_l)
                test_error_2 = test_error_2 + 1;
            end
            if(i_n_5-1 ~= test_l)
                test_error_5 = test_error_5 + 1;
            end
        end
         
        train_l = train_labels(i);
        
        c_label = labels(:,train_l+1);
        
        h_units_1 = w_1_1'*train_images(i,:)';
        h_units_2 = w_1_2'*train_images(i,:)';
        h_units_5 = w_1_5'*train_images(i,:)';
        
        r_units_1 = h_units_1;
        r_units_2 = h_units_2;
        r_units_5 = h_units_5;
        
        zero_1 = find(h_units_1 <= 0);
        zero_2 = find(h_units_2 <= 0);
        zero_5 = find(h_units_5 <= 0);

        for j = 1:length(zero_1)
            r_units_1(zero_1(j)) = 0;
        end
        for j = 1:length(zero_2)
            r_units_2(zero_2(j)) = 0;
        end
        for j = 1:length(zero_t)
            r_units_5(zero_t(j)) = 0;
        end
        
        v_1 = w_2_1'*r_units_1;
        v_2 = w_2_2'*r_units_2;
        v_5 = w_2_5'*r_units_5;

        y_1 = softmax(v_1);
        y_2 = softmax(v_2);
        y_5 = softmax(v_5);

        [~,i_t_1] = max(y_1);
        [~,i_t_2] = max(y_2);
        [~,i_t_5] = max(y_5);
        
        if(i_t_1-1 ~= train_l)
            train_error_1 = train_error_1 + 1;
        end
        if(i_t_2-1 ~= train_l)
            train_error_2 = train_error_2 + 1;
        end
        if(i_t_5-1 ~= train_l)
            train_error_5 = train_error_5 + 1;
        end  
        
        grad_2_1 = grad_2_1 + (c_label-y_1)*r_units_1';
        grad_2_2 = grad_2_2 + (c_label-y_2)*r_units_2';
        grad_2_5 = grad_2_5 + (c_label-y_5)*r_units_5';
        
        grad_1_1 = grad_1_1 + ((r_units_1.*(ones(10,1)-r_units_1)).*(w_2_1*(c_label-y_1)))*train_images(i,:);
        grad_1_2 = grad_1_2 + ((r_units_2.*(ones(20,1)-r_units_2)).*(w_2_2*(c_label-y_2)))*train_images(i,:);
        grad_1_5 = grad_1_5 + ((r_units_5.*(ones(50,1)-r_units_5)).*(w_2_5*(c_label-y_5)))*train_images(i,:);
        
    end
    
    error_1 = train_error_1/t_space_size(1);
    error_2 = train_error_2/t_space_size(1);
    error_3 = train_error_5/t_space_size(1);
    
    test_errors_1 = [test_errors_1; test_error_1/length(test_labels)];
    test_errors_2 = [test_errors_2; test_error_2/length(test_labels)];
    test_errors_5 = [test_errors_5; test_error_5/length(test_labels)];
    train_errors_1 = [train_errors_1; error_1];
    train_errors_2 = [train_errors_2; error_2];
    train_errors_5 = [train_errors_5; error_3];
    
    disp(['Error 1: ',num2str(error_1),' Error 2: ',num2str(error_2),' Error 3: ',num2str(error_3),' Time: ',num2str(t)]);
%     disp(['Error : ',num2str(error_2),' Time: ',num2str(t)]);
    if(error_1 < 0.1)
        flag = 1;
    else
        w_2_1 = w_2_1 + eta.*grad_2_1' - eta*lamba*w_2_1;
        w_2_2 = w_2_2 + eta.*grad_2_2' - eta*lamba*w_2_2;
        w_2_5 = w_2_5 + eta.*grad_2_5' - eta*lamba*w_2_5;
        
        w_1_1 = w_1_1 + eta.*grad_1_1' - eta*lamba*w_1_1;
        w_1_2 = w_1_2 + eta.*grad_1_2' - eta*lamba*w_1_2;
        w_1_5 = w_1_5 + eta.*grad_1_5' - eta*lamba*w_1_5;
    end

    grad_2_1 = zeros(10,10);
    grad_2_2 = zeros(10,20);
    grad_2_5 = zeros(10,50);

    grad_1_1 = zeros(10,t_space_size(2));
    grad_1_2 = zeros(20,t_space_size(2));
    grad_1_5 = zeros(50,t_space_size(2));
    t = t + 1;
    
end

%% Stochastic Gradient 1 Layer

t_space_size = size(train_images);

w = randn(t_space_size(2),10);
grad = zeros(10,t_space_size(2));
a = zeros(10,1);

test_errors = [];
train_errors = [];

t = 0;

labels = eye(10);

eta = 10^(-5);

flag = 0;

while(flag == 0)
    
    for i = 1:t_space_size(1)
        
        test_error = 0;
        train_error = 0;
        
        if(i <= length(test_labels))
            a = w'*test_images(i,:)';
            y = softmax(a);
            [~,i_n] = max(y);
            for j = 1:length(test_labels)
                test_l = train_labels(j);

                c_label = labels(:,test_l+1);

                if(i_n-1 ~= c_label(test_l+1))
                    test_error = test_error + 1;
                end
            end
        end
        
        a_t = w'*train_images(i,:)';
        
        y_t = softmax(a_t);
        
        [~,i_t] = max(y_t);
        
        for j = 1:t_space_size(1)
            train_l = train_labels(j);
        
            c_label = labels(:,train_l+1);
            
            if(i_t-1 ~= c_label(train_l+1))
                train_error = train_error + 1;
            end
        end
        
        for j = 1:10
            grad = grad + (labels(:,j)-y_t)*train_images(i,:);
        end
        
        test_errors = [test_errors; test_error/length(test_labels)];
        train_errors = [train_errors; train_error/t_space_size(1)];

        len_error = length(train_errors);
        if (len_error > 1)
            error = train_errors(len_error);
        else
            error = 10;
        end

        disp(['Error: ',num2str(train_errors(len_error)),' Time: ',num2str(t)]);
        if(error < 0.01)
            flag = 1;
        else
            w = w + eta.*grad';
        end

        grad = zeros(10,t_space_size(2));
        t = t + 1;
    end
    
end

%% Stochastic Descent 2 Layer Sigmoid & ReLU

