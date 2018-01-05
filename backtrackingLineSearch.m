function backtrackingLineSearch

x = 0;
t = 0.1;
maxIter = 30;
objValue = [];

for i = 1:maxIter
    [f,g] = objFun(x);
    objValue = [objValue;f];
    x = x - t*g;
end

hold on;
plot(objValue,'bo-');
hold off;

x = 0;
alpha = 0.5;
beta = 0.8;

objValue = [];
for i = 1:maxIter
    [f,g] = objFun(x);
    objValue = [objValue;f];
    t = 1;
    
    while 1
        considerx = x - t*g;
        [newf,newg] = objFun(considerx);
        if newf > f - alpha*t*g^2
            t = beta*t;
        else
            break;
        end
    end
    
    x = x - t*g;
end

hold on;
plot(objValue,'ro-');
hold off;
legend('Normal','Backtracking line search');
box on;
end

function [cost, grad] = objFun(x)
cost = (x-3)^2;
grad = 2*(x-3);
end