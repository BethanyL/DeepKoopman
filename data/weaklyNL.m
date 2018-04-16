function dy = weaklyNL(y,b,mu,omega,lambda,A)

dy = [
    mu*y(1) - omega*y(2) + A*y(1)*y(3);
    omega*y(1) + mu*y(2) + A*y(2)*y(3) + b;
    lambda*(y(3)-y(1).^2-y(2).^2);  
];