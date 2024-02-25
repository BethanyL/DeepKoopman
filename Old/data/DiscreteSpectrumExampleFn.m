function X = DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lambda, seed)

% Koopman example from 3.4.2 (pg 51) of the DMD book
% (Dynamic Mode Decomposition by Kutz, Brunton, Brunton, and Proctor)
% nonlinear dynamical system in two variables, but with 3D Koopman
% observables, have linear dynamical system

% try some initial conditions for x1, x2
rng(seed)

% randomly start from x1range(1) to x1range(2)
x1 = (x1range(2)-x1range(1))*rand([numICs,1])+x1range(1);

% randomly start from x2range(1) to x2range(2)
x2 = (x2range(2)-x2range(1))*rand([numICs,1])+x2range(1);

lenT = length(tSpan);

X = zeros(numICs*lenT, 2);

count = 1;
% in order to solve more accurately than ode45, map into 3D linear system
% and use exact analytic solution 
for j = 1:numICs
    Y0 = [x1(j); x2(j); x1(j)^2];
    c1 = Y0(1);
    c2 = Y0(2) + (lambda*Y0(3))/(2*mu-lambda);
    c3 = (-lambda*Y0(3))/(2*mu-lambda);
    c4 = Y0(3);
    Y = [c1 * exp(mu*tSpan);
        c2 * exp(lambda*tSpan) + c3 * exp(2*mu*tSpan);
        c4 * exp(2*mu*tSpan)];

    Xhat = Y(1:2,:);
    X(1+(count-1)*lenT : lenT + (count-1)*lenT,:) = Xhat(:,1:lenT)';
    count = count + 1;
end




