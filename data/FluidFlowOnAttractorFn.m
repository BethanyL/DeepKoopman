function X = FluidFlowOnAttractorFn(Rrange, Trange, numICs, tSpan, seed)


% try some initial conditions for R, T
rng(seed)

b = 0;
mu = 0.1;
omega = 1;
lambda = -10;
A = -mu;
dynsys = @(t,x) weaklyNL(x,b,mu,omega,lambda,A);

lenT = length(tSpan);

X = zeros(numICs*lenT, 3);

count = 1;
for j = 1:2*numICs
    % randomly start from Rrange(1) to Rrange(2)
    R = (Rrange(2)-Rrange(1))*rand+Rrange(1);

    % randomly start from Trange(1) to Trange(2)
    T = (Trange(2)-Trange(1))*rand+Trange(1);

    x1 = R*cos(T);
    x2 = R*sin(T);
    x3 = x1^2 + x2^2;

    ic = [x1; x2; x3];

    [T, temp] = ode45(dynsys, tSpan, ic);

    X(1+(count-1)*lenT : lenT + (count-1)*lenT,:) = temp;
    if count == numICs
        break
    end
    count = count + 1;
end



