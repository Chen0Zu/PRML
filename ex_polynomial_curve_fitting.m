close all;clc;clear;rng('default');
addpath('./fun');
%test;
%% ========================================================================
N = 10;
X = rand(N,1);
t = sin(2*pi*X) + 0.2*randn(N,1);
X1 = X;
t1 = t;

data = [0:0.01:1]';
hold on;
plot(data,sin(2*pi*(data)));
plot(X,t,'o');
hold off;

%% ========================================================================
M = [0 1 3 9];
figure;
for i = 1:length(M)
    m = M(i);
    x_hat =  polynomial(X,m);
    x_hat = [ones(N,1),x_hat];
    w = x_hat'*x_hat \ (x_hat'*t);
    disp(w);
    pause;
    data_hat = polynomial(data,m);
    data_hat = [ones(size(data_hat,1),1),data_hat];
    y = data_hat*w;
    subplot(2,2,i)
    hold on;
    plot(data,sin(2*pi*(data)));
    plot(X,t,'o');
    plot(data,y,'r-');
    axis([-0.1, 1.1 -1.3 1.3]);
    box on;
    hold off;
    title(['M = ',num2str(m)]);
end

%% ========================================================================
Xtest = rand(100,1);
ttest = sin(2*pi*Xtest) + 0.2*randn(100,1);
M = 0:8;
err_train = zeros(length(M),1);
err_test = zeros(length(M),1);
for i = 1:length(M)
    m = M(i);
    x_hat =  polynomial(X,m);
    x_hat = [ones(N,1),x_hat];
    w = x_hat'*x_hat \ (x_hat'*t);
    err_train(i) = sqrt(sum((t - x_hat*w).^2)/length(t));
    Xtest_hat = polynomial(Xtest,m);
    Xtest_hat = [ones(size(Xtest_hat,1),1),Xtest_hat];
    y_test = Xtest_hat*w;
    err_test(i) = sqrt(sum((ttest - y_test).^2)/length(ttest));
end
figure;
hold on;
plot(M,err_train,'o-r');
plot(M,err_test,'o-b');
hold off;
legend('Training','Test');

%% =======================================================================
N = 15;
X = rand(N,1);
t = sin(2*pi*X) + 0.2*randn(N,1);
x_hat =  polynomial(X,m);
x_hat = [ones(N,1),x_hat];
w = x_hat'*x_hat \ (x_hat'*t);
data_hat = polynomial(data,m);
data_hat = [ones(size(data_hat,1),1),data_hat];
y = data_hat*w;
figure;
subplot(1,2,1)
hold on;
plot(data,sin(2*pi*(data)));
plot(X,t,'o');
plot(data,y,'r-');
axis([-0.1, 1.1 -1.3 1.3]);
box on;
hold off;
title('N=15');

N = 100;
X = rand(N,1);
t = sin(2*pi*X) + 0.2*randn(N,1);
x_hat =  polynomial(X,m);
x_hat = [ones(N,1),x_hat];
w = x_hat'*x_hat \ (x_hat'*t);
data_hat = polynomial(data,m);
data_hat = [ones(size(data_hat,1),1),data_hat];
y = data_hat*w;
subplot(1,2,2)
hold on;
plot(data,sin(2*pi*(data)));
plot(X,t,'o');
plot(data,y,'r-');
axis([-0.1, 1.1 -1.3 1.3]);
box on;
hold off;
title('N=100');

%% ======================================================================
lambda = exp(-18);
m = 9;
x_hat =  polynomial(X1,m);
x_hat = [ones(length(x_hat),1),x_hat];
w = (x_hat'*x_hat +  lambda*eye(size(x_hat,2)))\ (x_hat'*t1);
data_hat = polynomial(data,m);
data_hat = [ones(size(data_hat,1),1),data_hat];
y = data_hat*w;
figure;
subplot(1,2,1)
hold on;
plot(data,sin(2*pi*(data)));
plot(X,t,'o');
plot(data,y,'r-');
axis([-0.1, 1.1 -1.3 1.3]);
box on;
hold off;
title('log(\lambda)=-18');
disp(w);
pause

lambda = exp(0);
x_hat =  polynomial(X1,m);
x_hat = [ones(length(x_hat),1),x_hat];
w = (x_hat'*x_hat +  lambda*eye(size(x_hat,2)))\ (x_hat'*t1);
data_hat = polynomial(data,m);
data_hat = [ones(size(data_hat,1),1),data_hat];
y = data_hat*w;
subplot(1,2,2)
hold on;
plot(data,sin(2*pi*(data)));
plot(X,t,'o');
plot(data,y,'r-');
axis([-0.1, 1.1 -1.3 1.3]);
box on;
hold off;
title('log(\lambda)=0');