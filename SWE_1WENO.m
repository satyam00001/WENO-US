function SWE_1WENO
IC = 2; % 0 = Lax, 1 = Shock-entropy wave interaction
N = 201; % Number of Grid points/cells
g = 9.812; % Specific heat ratio
cfl = 0.45 ; % CFL number
Order = 5 ; % WENO Order
epsilon = 10^(-16) ; % WENO sensitivity parameter
p = 2 ; % WENO power parameter
%--------------- END INPUT-----------------------------------------------------
r = (Order+1)/2;
Step = 0; Time = 0; 
[x, dx, Final_Time, Q] = Initial_Condition(IC, N); % Initial Condition
while Time < Final_Time
dt = cfl*dx/Maximum_Eigenvalue(Q);
if dt > (Final_Time - Time), dt = Final_Time - Time; end
%3rd order TVD Runge-Kutta scheme
Q1 = Q + dt * Diff(Q) ;
Q2 = (3*Q + Q1 + dt * Diff(Q1) )/4;
Q = ( Q + 2*Q2 + 2*dt * Diff(Q2) )/3;
Time = Time + dt;Step = Step + 1;
end 
figure;
subplot(2,1,1);
plot(x, Q(1,:), 'k-');
xlabel('x'); ylabel('h(x,t)'); title('Water Height');
legend('WENO-US5');

subplot(2,1,2);
plot(x, Q(2,:)./ Q(1,:), 'k-');
xlabel('x'); ylabel('u(x,t)'); title('Velocity');
legend('WENO-US5');
%------------------------Functions-----------------------
function [x, dx] = GetGrid(x0, x1, N)
dx = abs(x1-x0)/N;
x = x0:dx:x1;
end
function [x, dx, Final_Time, Q0] = Initial_Condition (IC, N)
switch IC % function for IC
case 0
[x, dx] = GetGrid(-1,1, N);
 Final_Time = 0.05;
        h0 = 2.0 * ones(size(x)); h0(x >= 0) = 1;  % Water height
        u0 = zeros(size(x));                        % Velocity
   case 1
    [x, dx] = GetGrid(-1, 1, N);
    Final_Time = 0.1;
    h0 = 0.1 * ones(size(x));                  % Water height
    u0 = 0 * ones(size(x)); 
    u0(x >= 0) = 3;                            % Velocity
      

end
Q0 = [h0; h0 .* u0];                         % [h; hu]
end
function Qa = Average(Q)
r =3;
Qa = (Q(:,r) + Q(:,r+1))/2;
end
%------------------------Euler equations----------------------
function [R, L, Lambda] = Eigensystem(Q)
    h = Q(1); u = Q(2)./h;
    a = sqrt(g*h);
    R = [1, 1;
         u - a, u + a];
     L = (1 / (2 * a)) * [u + a, -1;
                         -(u - a), 1];
    Lambda = [u - a; u + a];
end

function Lambda = Eigenvalues(Q)
    h = Q(1,:);
    u = Q(2,:) ./ h;
    a = sqrt(g * h); % gravity wave speed
    Lambda = [u - a; u + a];
end

function Lambda_Max = Maximum_Eigenvalue(Q)
    Lambda_Max = max( max( abs(Eigenvalues(Q)) ) );
end
%------------------------Boundary Condition----------------------
function f_BC = BoundaryCondition(f)
f_BC = [f(:,1) f(:,1) f(:,1) f f(:,end) f(:,end) f(:,end)]; %zero gradient
end
%------------------------WENO Differentiation--------------------
function D_F = Diff(Q)
Q = BoundaryCondition(Q);
NN = size(Q,2); F_half = zeros(2, NN-(2*r-1)); G_half = zeros(2, 1);
alpha = Maximum_Eigenvalue(Q);
for i = 1:NN-(2*r-1)
Qi = Q(:, i:i+5);
Lambda = Eigenvalues(Qi);
Qa = Average(Qi);
[R, L] = Eigensystem(Qa);
W = L*Qi ; % Transforms into characteristic variables
G = Lambda.* W; % The flux for the characteristic variables is Lambda * L*Q
for j = 1:2 % WENO reconstruction of the flux G
G_half(j) = WENO_Flux_at_Half_Point(G(j,:), W(j,:), alpha);
end
F_half(:,i) = R*G_half; % Brings back to conservative variables
end
D_F = -(F_half(:,2:end) - F_half(:,1:end-1))/dx; % -Derivative of Flux
end
function F_half = WENO_Flux_at_Half_Point(F, Q, alpha)
F_plus = (F + alpha*Q)/2;
F_minus = (F - alpha*Q)/2;
%--------------------------------------------------------
% WENO Reconstruction at cell boundary
F_half_plus = WENO_Reconstruction(F_plus );
% The number of ghost cells must be the SAME in the right and left boundaries
G_minus = F_minus(end:-1:1);
G_minus = WENO_Reconstruction(G_minus);
F_half_minus = G_minus(end:-1:1);
%--------------------------------------------------------
F_half = F_half_plus + F_half_minus;
F_half = F_half(:,r+1);
end
%------------------------WENO Reconstruction---------------------
function f_half = WENO_Reconstruction(f)
    r = (Order+1)/2 ; NN = length(f); f_half = zeros(1,NN+1);
    d = [1/10, 6/10, 3/10]; c1 = 1/4 ; c2 = 13/12 ;% Ideal Weights
    s2 = 1/12 ;
    s1 = 1/4;
    f = f(:) ; % Makes f a column vector
    P = zeros(3,1); % P is a column vector
    beta = zeros(1,3);
    R = zeros(1,3);
    S = zeros(1,3);
    
    for i = r: NN-r
   % lower order polynomial in substencil S^k, k=0,1,2 using cell center value f_i
            P(1) = [ 2 -7 11]/6 * f(i-2:i );
            P(2) = [-1 5 2]/6 * f(i-1:i+1);
            P(3) = [ 2 5 -1]/6 * f( i:i+2);
    % local lower order smoothness indicator in substencil S^k, k=0,1,2
        beta(1) = c1*([ 1 -4 3] * f(i-2:i )).^2 + c2*([1 -2 1] * f(i-2:i )).^2;
        beta(2) = c1*([-1 0 1] * f(i-1:i+1)).^2 + c2*([1 -2 1] * f(i-1:i+1)).^2;
        beta(3) = c1*([-3 4 -1] * f( i:i+2)).^2 + c2*([1 -2 1] * f( i:i+2)).^2;
        b(1) = ([ 1 -1  0] * f(i-2:i )).^2 ;
        b(2) = ([1 -1 0] * f(i-1:i+1)).^2 ;
        b(3) = ([1 -1 0] * f( i:i+2)).^2 ;    
        b(4) = ([0 1 -1] * f( i:i+2)).^2 ;
        b(5) = 1/4 * ([-1 0 1] * f(i-1:i+1)).^2;
        b(6) = 1/4 * ([1 -4 3] * f(i-2:i)).^2;
        b(7) = 1/4 * ([3 -4 1] * f(i:i+2)).^2;
        B = zeros(1,5);
        B(4) = 1/36 * ([0 2 3 -6 1]* f(i-2:i+2)).^2;
        B(5) = 1/144 * ([1 -8 0 8 -1]* f(i-2:i+2)).^2;
        % WENO-JS weights: alpha_k
        alpha = d./(beta + epsilon).^p;
        %WENO-Z weights: alpha_k
        %tau = abs(beta(3) - beta(1));
        tau= (abs(b(6)-B(5)) + abs(b(7)-B(5)) -2* abs(b(5)-B(5)));
        alpha = d.*( 1 + E.*(tau./( beta + epsilon) ).^p );
        %alpha = d.*( 1 +( tau./( beta + epsilon) ).^p );
        % Nonlinear weights: omega_k
        omega = alpha/sum(alpha);
       % Reconstructed polynomial f_half at the cell boundary x_(i+1/2)
        f_half(i+1) = omega*P;
    end
end

end
