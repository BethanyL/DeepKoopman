function X = FluidFlowBoxFn(x1range, x2range, x3range, numICs, tSpan, seed, max_x3)


% try some initial conditions for x1, x2
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

for j = 1:10*numICs
    % randomly start from x1range(1) to x1range(2)
    x1 = (x1range(2)-x1range(1))*rand+x1range(1);

    % randomly start from x2range(1) to x2range(2)
    x2 = (x2range(2)-x2range(1))*rand+x2range(1);

    % randomly start from x3range(1) to x3range(2)
    x3 = (x3range(2)-x3range(1))*rand+x3range(1);


    ic = [x1; x2; x3];

    [T, temp] = ode45(dynsys, tSpan, ic);

    if max(temp(:,3)) > max_x3
        sprintf('traj goes too big: %.15f', max(temp(:,3)))
        continue
    else
        X(1+(count-1)*lenT : lenT + (count-1)*lenT,:) = temp;

        if count == numICs
            break
        end
        count = count + 1;
    end
end



