function SWE_DamBreak_JA
% SWE_2D
% Dam_breaking
% ====================================================
x0 = 0 ;
x1 = 200 ;
N_x = 201 ; % number of points
% ====================================================
y0 = 0 ;
y1 = 200 ;
N_y = 201 ; % number of points
% ====================================================
Gravity = 9.812 ;
Final_Time = 7.2;
CFL = 0.45 ;
Movie_Resolution = 0.1 ;
% ====================================================
NV = 3 ;
% ====================================================
% WENO Setup
WENO_Order = 5 ;
WENO_Flag = 1;
WENO_Epsilon = 1e-16 ;
WENO_Power = 2 ;
% ====================================================
% Index Setup
r = (WENO_Order + 1)/2 ;
N0 = 1 ;
N2 = N0 + r ;
N1 = N2 - 1 ;
N3 = N2 + (N_x-1) - 1 ;
N4 = N3 + 1 ;
N5 = N3 + r ;
M0 = 1 ;
M2 = M0 + r ;
M1 = M2 - 1 ;
M3 = M2 + (N_y-1) - 1 ;
M4 = M3 + 1 ;
M5 = M3 + r ;
% ====================================================
[ X , dx ] = Domain_Setup ( x0, x1, N_x, N0, N2, N3, N5 ) ;
[ Y , dy ] = Domain_Setup ( y0, y1, N_y, M0, M2, M3, M5 ) ;
[ x, y ] = Mesh_Setup ( X, Y );
[Ix0, Ix1, Iy0, Iy1] = Obstacle_Find( X, Y );
[Q, bot] = Initial_Condition ( x, y ) ;
Q = Boundary_Condition ( Q ) ;
Q0 =Q;
Step = 0; Time = 0; dt = Time_Step ( Q ) ;
% ====================================================
disp(['Estimate Number of Time Step : ', num2str(fix((Final_Time-Time)/dt))])
disp(' Total_Step Step dt Time CPU Time')
disp('--------------------------------------------------------------------------')
formatstr = ' %7d %7d %13.4E %13.4E %13.4E %13.4E %13.4E';
fprintf(formatstr, fix((Final_Time-Time)/dt), Step, dt, Time, 0, 0)
fprintf('\n')
% ====================================================
Next_Save_Time = Time + Movie_Resolution;
CPU_Begin = cputime;
while (Time < Final_Time)
dt = Time_Step ( Q ) ;
Stability_Check
Step = Step + 1 ; Time = Time + dt ;
Save_Indicator = 0;
if (Time >= Next_Save_Time)
dt = dt - (Time-Next_Save_Time);
Time = Next_Save_Time ;
Save_Indicator = 1;
Next_Save_Time = Next_Save_Time + Movie_Resolution;
end
if Time > Final_Time
dt = dt - (Time - Final_Time);
Time = Final_Time ;
end
CPU_Start = cputime ;
Q = Runge_Kutta ( Q ) ;
CPU_End = cputime ;
%if (Save_Indicator)
%fprintf('Plot ... \n'); 
%contour_levels = [5.2, 5.7, 6.2, 6.7, 7.2, 7.8, 8.2, 8.7, 9.2];
%H = Q(:,:,1);  % Water height
%figure;
%contour(X, Y, H', contour_levels, 'ShowText','on');
%colorbar;
%title('Water Surface Elevation Contours');
%xlabel('x (m)');
%ylabel('y (m)');
%if (Save_Indicator)
%printf('Plot ... \n') ; 
%mesh(x,y, Q(:,:,1)); grid on
%title(['t = ', num2str(Time, '%1.2f')]);
%xlim( [x0, x1] )
%ylim( [y0, y1] )
%pause( 0.001 )
%end
%end
% === ERROR ANALYSIS ===
% === ERROR ANALYSIS ===
H = Q(:,:,1);         % Final solution
h_ref = Q0(:,:,1);    % Initial profile

err = abs(H - h_ref);
L1_err  = mean(err(:));
L2_err  = sqrt(mean(err(:).^2));
Linf_err = max(err(:));

fprintf('\nERROR NORMS (vs initial condition):\n');
fprintf('  L1 Norm     = %.4e\n', L1_err);
fprintf('  L2 Norm     = %.4e\n', L2_err);
fprintf('  L-infinity  = %.4e\n', Linf_err);

fprintf(formatstr, fix((Final_Time-Time)/dt), Step, dt, ...
Time, CPU_End-CPU_Start, CPU_End-CPU_Begin)
fprintf('\n')
end
fprintf('Done ... \n');
% ================================================================
function [x, dx] = Domain_Setup ( x0, x1, N, N0, N2, N3, N5 )
dx = (x1-x0)/(N-1);
x = x0 + ((N0:N5)-N2)*dx + 1/2*dx ;
end
% ================================================================
function [x, y] = Mesh_Setup ( X, Y )
x = zeros( N5, M5 );
y = zeros( N5, M5 );
for i = N0:N5
y(i,:) = X(:);
end
for j = M0:M5
x(:,j) = Y;
end
end
% ================================================================
function Q = Runge_Kutta ( Q )
% Stage 1
t = Time ;
D_F = D_Fluxes( Q , bot ) ;
Q1 = Q + dt * D_F ;
Q1 = Boundary_Condition ( Q1 ) ;
% Stage 2
t = Time - dt/2 ;
D_F = D_Fluxes( Q1, bot ) ;
Q1 = (3.0*Q + Q1 + dt*D_F)/4.0 ;
Q1 = Boundary_Condition ( Q1 ) ;
% Stage 3
t = Time + dt ;
D_F = D_Fluxes( Q1, bot ) ;
Q = (Q + 2*Q1 + 2*dt*D_F)/3.0 ;
Q = Boundary_Condition ( Q ) ;
end
% ================================================================
function dt = Time_Step ( Q )
H = Q(:,:,1);
U = Q(:,:,2)./H;
V = Q(:,:,3)./H;
C = sqrt(abs(Gravity * H ));
U_Max = max( abs(U(:)) + C(:) );
V_Max = max( abs(V(:)) + C(:) );
dt = CFL * (dx/U_Max + dy/V_Max);
end
% ================================================================
function D_F = D_Fluxes ( Q, bot )
D_F_x = zeros(N5,M5,NV);
D_F_y = zeros(N5,M5,NV);
D_S_x = zeros(N5,M5,NV);
D_S_y = zeros(N5,M5,NV);
Index_x = 1;
Index_y = 2;
Q = Int_Boundary_x ( Q );
for j = M0:M5
[D_F_x(:,j,:), D_S_x(:,j,:)] = D_Fluxes_XY ( N2, N3, N5, ...
Q(:,j,:), bot(:,j), Index_x, dx );
end
Q = Int_Boundary_y ( Q );
for i = N0:N5
[D_F_y(i,:,:), D_S_y(i,:,:)] = D_Fluxes_XY ( M2, M3, M5, ...
Q(i,:,:), bot(i,:), Index_y, dy );
end
D_F = -(D_F_x + D_F_y) + (D_S_x + D_S_y);
end
% ================================================================
function [D_F, D_S] = D_Fluxes_XY ( N2, N3, N5, Q, bot, Index, dx )
half = 1/2;
Q = squeeze( Q );
bot = bot(:);
D_F = zeros(N5,NV);
D_S = zeros(N5,NV);
F = zeros(N5,NV);
Fh = zeros(N5,NV);
Sh = zeros(N5,NV);
H = Q(:,1);
U = Q(:,2)./H;
V = Q(:,3)./H;
P = half * Gravity * H.^2;
C = sqrt( abs(Gravity * H) );
if Index == 1
F(:,1) = Q(:,1).*U;
F(:,2) = Q(:,2).*U + P - half * Gravity * bot.^2;
F(:,3) = Q(:,3).*U;
Lambda = max( abs(U(:)) + C(:) );
else
F(:,1) = Q(:,1).*V;
F(:,2) = Q(:,2).*V;
F(:,3) = Q(:,3).*V + P - half * Gravity * bot.^2;
Lambda = max( abs(V(:)) + C(:) );
end
S = bot;
Q1 = Q;
Q1(:,1) = H + bot;
F_P = half * (F + Lambda * Q1);
F_M = half * (F - Lambda * Q1);
S_P = half * S;
S_M = half * S;
for i = N2-1:N3
for k = 1:NV
if k ~= Index + 1
Fh_P = WENO5_P ( F_P( i-2:i+3,k ) );
Fh_M = WENO5_M ( F_M( i-2:i+3,k ) );
else
[Fh_P, Sh_P] = WENO5_P2 ( F_P(i-2:i+3,k), S_P(i-2:i+3) );
[Fh_M, Sh_M] = WENO5_M2 ( F_M(i-2:i+3,k), S_M(i-2:i+3) );
Sh(i,k) = Sh_P + Sh_M;
end
Fh(i,k) = Fh_P + Fh_M;
end
end
D_F(N2:N3,:) = (Fh(N2:N3,:) - Fh(N2-1:N3-1,:))/dx;
D_S(N2:N3,Index + 1) = - Gravity * ( H(N2:N3) + bot(N2:N3) ) .* ...
(Sh(N2:N3,Index + 1) - Sh(N2-1:N3-1,Index + 1))/dx;
end
% ================================================================
function fhalf = WENO5_P( f )
r = (WENO_Order + 1)/2 ;
ep = WENO_Epsilon ;
p = WENO_Power ;
flag = WENO_Flag ;
d = [1/10, 6/10, 3/10];
C1 = 1/4; C2 = 13/12;
f = f(:);
P = zeros(r,1);
beta = zeros(1,r);
%P(1) = [ 3 -10 15]/8 * f(r-2:r );
%P(2) = [-1 6 3]/8 * f(r-1:r+1);
%P(3) = [ 3 6 -1]/8 * f( r:r+2);
P(1) = [ 2, -7, 11]/6 * f(r-2:r );
P(2) = [-1, 5, 2]/6 * f(r-1:r+1);
P(3) = [ 2, 5, -1]/6 * f(r :r+2);
beta(1) = C1 * ([ 1, -4, 3] * f(r-2:r )).^2 + ...
C2 * ([ 1, -2, 1] * f(r-2:r )).^2;
beta(2) = C1 * ([-1, 0, 1] * f(r-1:r+1)).^2 + ...
C2 * ([ 1, -2, 1] * f(r-1:r+1)).^2;
beta(3) = C1 * ([-3, 4, -1] * f(r :r+2)).^2 + ...
C2 * ([ 1, -2, 1] * f(r :r+2)).^2;
%beta(1) = 1/2*(([ -1 1 0] * f(r-2:r )).^2 + ([0 -1 1] * f(r-2:r )).^2) + ([1 -2 1] * f(r-2:r)).^2;
%beta(2) = 1/2*(([ -1 1 0] * f(r-1:r+1 )).^2 + ([0 -1 1] * f(r-1:r+1)).^2) + ([1 -2 1] * f(r-1:r+1)).^2;
%beta(3) = 1/2*(([ -1 1 0] * f(r:r+2 )).^2 + ([0 -1 1] * f(r:r+2 )).^2) + ([1 -2 1] * f(r:r+2)).^2;
b(5) = 1/4 * ([-1 0 1] * f(r-1:r+1)).^2;
b(6) = 1/4 * ([1 -4 3] * f(r-2:r)).^2;
b(7) = 1/4 * ([3 -4 1] * f(r:r+2)).^2;
B(5) = 1/144 * ([1 -8 0 8 -1]* f(r-2:r+2)).^2;
%E = [1/10,1/10,4/5];
if flag == 1
% WENO-Z weights : alpha_k
tau = abs(beta(3) - beta(1));
%tau = abs(-2*([1 -2 1] * f(r-1:r+1)).^2 +  ([1 -2 1] * f(r-2:r)).^2 +([1 -2 1] * f(r:r+2)).^2);
%tau= (abs(b(6)-B(5)) + abs(b(7)-B(5)) -2* abs(b(5)-B(5)));
alpha = d .* (1 + (tau ./ (ep+beta)).^p);
else
alpha = d ./ (ep+beta).^p;
end
omega = alpha / sum(alpha);
fhalf = omega * P;
end
% ================================================================
function fhalf = WENO5_M( f )
fhalf = WENO5_P( f(end:-1:1) );
end
% ================================================================
function [fhalf, shalf] = WENO5_P2( f, s )
r = (WENO_Order + 1)/2 ;
ep = WENO_Epsilon ;
p = WENO_Power ;
flag = WENO_Flag ;
d = [1/10, 6/10, 3/10];
C1 = 1/4; C2 = 13/12;
f = f(:);
P = zeros(r,1);
S = zeros(r,1);
beta = zeros(1,r);
%P(1) = [ 3 -10 15]/8 * f(r-2:r );
%P(2) = [-1 6 3]/8 * f(r-1:r+1);
%P(3) = [ 3 6 -1]/8 * f( r:r+2);
P(1) = [ 2, -7, 11]/6 * f(r-2:r );
P(2) = [-1, 5, 2]/6 * f(r-1:r+1);
P(3) = [ 2, 5, -1]/6 * f(r :r+2);
%S(1) = [ 2, -7, 11]/6 * s(r-2:r );
%S(2) = [-1, 5, 2]/6 * s(r-1:r+1);
%S(3) = [ 2, 5, -1]/6 * s(r :r+2);
beta(1) = C1 * ([ 1, -4, 3] * f(r-2:r )).^2 + ...
C2 * ([ 1, -2, 1] * f(r-2:r )).^2;
beta(2) = C1 * ([-1, 0, 1] * f(r-1:r+1)).^2 + ...
C2 * ([ 1, -2, 1] * f(r-1:r+1)).^2;
beta(3) = C1 * ([-3, 4, -1] * f(r :r+2)).^2 + ...
C2 * ([ 1, -2, 1] * f(r :r+2)).^2;
%beta(1) = 1/2*(([ -1 1 0] * f(r-2:r )).^2 + ([0 -1 1] * f(r-2:r )).^2) + ([1 -2 1] * f(r-2:r)).^2;
%beta(2) = 1/2*(([ -1 1 0] * f(r-1:r+1 )).^2 + ([0 -1 1] * f(r-1:r+1)).^2) + ([1 -2 1] * f(r-1:r+1)).^2;
%beta(3) = 1/2*(([ -1 1 0] * f(r:r+2 )).^2 + ([0 -1 1] * f(r:r+2 )).^2) + ([1 -2 1] * f(r:r+2)).^2;
b(5) = 1/4 * ([-1 0 1] * f(r-1:r+1)).^2;
b(6) = 1/4 * ([1 -4 3] * f(r-2:r)).^2;
b(7) = 1/4 * ([3 -4 1] * f(r:r+2)).^2;
B(5) = 1/144 * ([1 -8 0 8 -1]* f(r-2:r+2)).^2;
%E = [1/10,1/10,4/5];
if flag == 1
tau = abs(beta(3) - beta(1));
%tau = abs(beta(3) - beta(1));
%tau = abs(-2*([1 -2 1] * f(r-1:r+1)).^2 +  ([1 -2 1] * f(r-2:r)).^2 +([1 -2 1] * f(r:r+2)).^2);
%tau= (abs(b(6)-B(5)) + abs(b(7)-B(5)) -2* abs(b(5)-B(5)));
alpha = d .* (1 + (tau ./ (ep+beta)).^p);
else
alpha = d ./ (ep+beta).^p;
end
omega = alpha / sum(alpha);
fhalf = omega * P;
shalf = omega * S;
end
% ================================================================
function [fhalf, shalf] = WENO5_M2( f, s )
[fhalf, shalf] = WENO5_P2( f(end:-1:1), s(end:-1:1) );
end
% ================================================================
function [Q, bot] = Initial_Condition( x, y )
Q = zeros(N5,M5,NV);
H = zeros(N5,M5);
U = zeros(N5,M5);
V = zeros(N5,M5);
bot = zeros(N5,M5);
xc = 100;
for i = N0:N5
for j = M0:M5
H(i,j) = 5;
if (x(i,j) < xc)
H(i,j) = 10 - bot(i,j);
end
end
end
Q(:,:,1) = H;
Q(:,:,2) = H.*U;
Q(:,:,3) = H.*V;
end
% ================================================================
function [Ix0, Ix1, Iy0, Iy1] = Obstacle_Find( X, Y )
bx0 = 97;
bx1 = 102;
by0 = 95;
by1 = 170;
for i = N2:N3
if ( X(i) <= bx0 && X(i+1) > bx0 )
Ix0 = i;
continue;
end
if ( X(i) <= bx1 && X(i+1) > bx1 )
Ix1 = i;
break;
end
end
for j = M2:M3
if ( Y(j) <= by0 && Y(j+1) > by0 )
Iy0 = j;
continue;
end
if ( Y(j) <= by1 && Y(j+1) > by1 )
Iy1 = j;
break;
end
end
end
% ================================================================
function Q = Boundary_Condition( Q )
    for k = 1:NV
       for i = N0:N1
         Q(i,:,k) = Q(N2,:,k);
       end
    for i = N4:N5
       Q(i,:,k) = Q(N3,:,k);
    end
    end
L1 = M1 - M0;
L2 = M5 - M4;
Q(:,M0:M1,:) = Q(:,M2+L1:-1:M2,:);
Q(:,M0:M1,3) = -Q(:,M2+L1:-1:M2,3);
Q(:,M4:M5,:) = Q(:,M3:-1:M3-L2,:);
Q(:,M4:M5,3) = -Q(:,M3:-1:M3-L2,3);
Q = Obstacle_height_setting( Q );
end
% ================================================================
function Q = Int_Boundary_x ( Q )
for j = M0:M5
if (Y(j) <= Y(Iy0) || Y(j) >= Y(Iy1))
Q(Ix0 :Ix0+2,j,:) = Q(Ix0-1:-1:Ix0-3, j,:);
Q(Ix0 :Ix0+2,j,2) = -Q(Ix0-1:-1:Ix0-3, j,2);
Q(Ix1-2:Ix1 ,j,:) = Q(Ix1+3:-1:Ix1+1, j,:);
Q(Ix1-2:Ix1 ,j,2) = -Q(Ix1+3:-1:Ix1+1, j,2);
end
end
end
% ================================================================
function Q = Int_Boundary_y ( Q )
for i = N0:N5
if (X(i) <= X(Ix1) && X(i) >= X(Ix0))
Q(i,Iy1 :Iy1+2,:) = Q(i,Iy1-1:-1:Iy1-3,:);
Q(i,Iy1 :Iy1+2,3) = -Q(i,Iy1-1:-1:Iy1-3,3);
Q(i,Iy0-2:Iy0 ,:) = Q(i,Iy0+3:-1:Iy0+1,:);
Q(i,Iy0-2:Iy0 ,3) = -Q(i,Iy0+3:-1:Iy0+1,3);
end
end
end
% ================================================================
function Q = Obstacle_height_setting( Q )
for i = Ix0:Ix1
for j = M0:M5
if (Y(j) <= Y(Iy0) || Y(j) >= Y(Iy1))
Q(i,j,1) = 10.2;
Q(i,j,2) = 0;
Q(i,j,3) = 0;
end
end
end
end
% ================================================================
function Stability_Check
if dt < 1.0e-16, error('========= STOP ! Unstable =============='); end
end 
end 
