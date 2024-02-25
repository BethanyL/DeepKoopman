function X = PendulumFn(x1range, x2range, numICs, tSpan, seed, max_potential)


% try some initial conditions for x1, x2
rng(seed)

dynsys = @(t,x) [x(2,:); -sin(x(1,:))];

lenT = length(tSpan);

X = zeros(numICs*lenT, 2);

potential = @(x,y) (1/2)*y^2-cos(x);

count = 1;
for j = 1:100*numICs
    % randomly start from x1range(1) to x1range(2)
    x1 = (x1range(2)-x1range(1))*rand+x1range(1);

    % randomly start from x2range(1) to x2range(2)
    x2 = (x2range(2)-x2range(1))*rand+x2range(1);


    if potential(x1, x2) <= max_potential
        ic = [x1; x2];

        [T, temp] = ode45(dynsys, tSpan, ic);

        X(1+(count-1)*lenT : lenT + (count-1)*lenT,:) = temp;
        if count == numICs
            break
        end
        count = count + 1;
    end
end

if count < numICs
    sprintf('oops, potential energy too small for IC box')
end


