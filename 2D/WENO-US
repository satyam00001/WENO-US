clear; %close all; clc;
global gamma 
%% Parameters
CFL = 0.45; %CFL number
tEnd = 0.3; %Final time
nx = 201; %Number of cells/Elements in x
ny = 201; %Number of cells/Elements in y
n = 5; % Degrees of freedom: ideal air=5, monoatomic gas=3.
IC = 1; % 2 IC cases are available
fspltMth='LF'; %LF, LLF.
reconMth='WENO5'; %WENO5, WENO7, Poly5, Poly7;
plotFig = false; %Visualize evolution of domain
% Ratio of specific heats for ideal di-atomic gas
gamma=(n+2)/n;
% Discretize spatial domain
Lx=1; dx=Lx/(nx-1); xc=0:dx:Lx;
Ly=1; dy=Ly/(ny-1); yc=0:dy:Ly;
[x,y] = meshgrid(xc,yc);
% Set IC
[r0,u0,v0,p0] = Euler_Riemann_IC2d(x,y,IC);
E0 = p0./(gamma-1)+0.5*r0.*(u0.^2+v0.^2); % Total Energy
c0 = sqrt(gamma*p0./r0); % Speed of sound
Q0 = cat(3, r0, r0.*u0, r0.*v0, E0); % initial state
% Set q-array & adjust grid for ghost cells
switch reconMth
 case {'WENO5','Poly5'}, R=3; nx=nx+2*R; ny=ny+2*R; in=R+1:ny-R; jn=R+1:nx-R;
	case {'WENO7','Poly7'}, R=4; nx=nx+2*R; ny=ny+2*R; in=R+1:ny-R; jn=R+1:nx-R;
end 
q0=zeros(ny,nx,4); q0(in,jn,:)=Q0;

% ENFORCED TIMESCALE RELATION: t = CFL * dx
dt0 = CFL * dx; 

% Discretize time domain
%vn = sqrt(u0.^2+v0.^2); lambda1=vn+c0; lambda2=vn-c0;
%a0 = max(abs([lambda1(:);lambda2(:)]));
%dt0=CFL*min(dx./a0,dy./a0);

% Initialize parpool
% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj); parpool('local',4); end

% Configure figure
%if plotFig
%figure(1);
%[~,h1]=contourf(x,y,r0); axis('square'); xlabel('x'); ylabel('y'); title('\rho');
%subplot(2,2,2); [~,h2]=contourf(x,y,u0); axis('square'); xlabel('x'); ylabel('y'); title('u_x');
%subplot(2,2,3); [~,h3]=contourf(x,y,v0); axis('square'); xlabel('x'); ylabel('y'); title('u_y');
%subplot(2,2,4); [~,h4]=contourf(x,y,p0); axis('square'); xlabel('x'); ylabel('y'); title('p');
%end

% Select Solver
solver = 1;
switch solver
 case 1, FD_EE2d = @FD_WENO_EE2d; % Component-wise reconstruction
 case 2, FD_EE2d = @FD_WENO_EE2d_CharactRecon; % Characteristic-wise reconstruction
end

%% Benchmark Loop: 100 Times Average
numRuns = 100;
totalCpuTime = 0;

fprintf('Running benchmark loop %d times...\n', numRuns);
for run = 1:numRuns
    % Load IC
    q = q0; t = 0; it = 0; dt = dt0; vn = sqrt(u0.^2+v0.^2); lambda1=vn+c0; lambda2=vn-c0; a = max(abs([lambda1(:);lambda2(:)]));
    
    tic
    while t < tEnd
     % Interaction local time
     if t+dt>tEnd; dt=tEnd-t; end; t=t+dt;
     
     % RK Initial step
     qo = q;
     
     % 1st stage
     L=FD_EE2d(q,a,nx,ny,dx,dy,t,fspltMth,reconMth,'Riemann'); q=qo-dt*L;
     
     % 2nd Stage
     L=FD_EE2d(q,a,nx,ny,dx,dy,t,fspltMth,reconMth,'Riemann'); q=0.75*qo+0.25*(q-dt*L);
     
     % 3rd stage
     L=FD_EE2d(q,a,nx,ny,dx,dy,t,fspltMth,reconMth,'Riemann'); q=(qo+2*(q-dt*L))/3;
     
     % Compute flow properties
     r=q(:,:,1); u=q(:,:,2)./r; v=q(:,:,3)./r; E=q(:,:,4); p=(gamma-1)*(E-0.5*r.*(u.^2+v.^2)); 
     c=sqrt(gamma*p./r);
     
     % Update wave speed parameter for flux splitting
     vn=sqrt(u.^2+v.^2); lambda1=vn+c; lambda2=vn-c;
     a = max(abs([lambda1(:);lambda2(:)])); 
     
     % ENFORCED TIMESCALE RELATION PRESERVED ACROSS STEPS
     dt = dt0; 
     
     % update iteration counter 
     it = it+1;
    end
    % Plot figure
    %plotFig && rem(it,10) == 0
    %set(h1,'ZData',r(in,jn));
    %set(h2,'ZData',u(in,jn));
    %set(h3,'ZData',v(in,jn));
    %set(h4,'ZData',p(in,jn));
    %drawnow
    %end
    totalCpuTime = totalCpuTime + toc;
end

averageCpuTime = totalCpuTime / numRuns;
disp(['Average CPU time over ', num2str(numRuns), ' runs: ', num2str(averageCpuTime), ' s']);

% Remove ghost cells for visualization calculations
q=q(in,jn,:); nx=nx-2*R; ny=ny-2*R; 
% compute flow properties
r=q(:,:,1); u=q(:,:,2)./r; v=q(:,:,3)./r; E=q(:,:,4); p=(gamma-1)*(E-0.5*r.*(u.^2+v.^2));
%% Calculation of flow parameters
c = sqrt(gamma*p./r); % Speed of sound
Mx = u./c; My = v./c; U = sqrt(u.^2+v.^2); M = U./c;
p_ref = 101325; % Reference air pressure (N/m^2)
rho_ref= 1.225; % Reference air density (kg/m^3)
s = 1/(gamma-1)*(log(p/p_ref)+gamma*log(rho_ref./r)); 
 % Entropy w.r.t reference condition
ss = log(p./r.^gamma); % Dimensionless Entropy
r_x = r.*u; % Mass Flow rate per unit area
r_y = r.*v; % Mass Flow rate per unit area
e = p./((gamma-1)*r); % internal Energy

%% Final plot
%n_lines=30; % contour lines
%contour(x,y,r,n_lines); axis('square'); xlabel('x(m)'); ylabel('Density (kg/m^3)');

function [r_0,u_0,v_0,p_0] = Euler_Riemann_IC2d(x,y,input)
%% Initial Physical Properties per case:
switch input
 case{1} % Test 1
 p = [ 1.0 1.0 1.0 1.0 ];
 r = [ 1.0 2.0 1.0 3.0 ];
 u = [-0.75 -0.75 0.75 0.75];
 v = [-0.5 0.5 0.5 -0.5 ];
 
 case{2} % Test 2
 p = [1.0 1.0 0.4 0.4 ];
 r = [1.0 2.0 1.0625 0.5197];
 u = [0.0 0.0 0.0 0.0 ];
 v = [1.0 -0.3 0.2145 0.2741];
 
 otherwise
 error('only 2 cases are available');
end

reg1 = (x>=0.5 & y>=0.5); 
reg2 = (x <0.5 & y>=0.5); 
reg3 = (x <0.5 & y <0.5); 
reg4 = (x>=0.5 & y <0.5); 

r_0 = r(1)*reg1 + r(2)*reg2 + r(3)*reg3 + r(4)*reg4; 
u_0 = u(1)*reg1 + u(2)*reg2 + u(3)*reg3 + u(4)*reg4; 
v_0 = v(1)*reg1 + v(2)*reg2 + v(3)*reg3 + v(4)*reg4; 
p_0 = p(1)*reg1 + p(2)*reg2 + p(3)*reg3 + p(4)*reg4; 
end 

function res = FD_WENO_EE2d(q,a,nx,ny,dx,dy,t,fsplitMth,Recon,Test)
 switch Recon
 case {'WENO5','Poly5'}, R=3;
 case {'WENO7','Poly7'}, R=4;
 otherwise, error('reconstruction not available');
 end
 
 switch Test
 case 'Smooth'
 for i=1:R
 q(:,i,:)=q(:,nx-R+i,:); q(:,nx-2*R+i,:)=q(:,R+i,:);
 end
 for j=1:R
 q(j,:,:)=q(ny-R+i,:,:); q(ny-2*R+j,:,:)=q(R+j,:,:);
 end
 case 'Riemann'
 for i=1:R
 q(:,i,:)=q(:,R+1,:); q(:,nx+1-i,:)=q(:,nx-R,:);
 end
 for j=1:R
 q(j,:,:)=q(R+1,:,:); q(ny+1-j,:,:)=q(ny-R,:,:);
 end
 end

 ic=R+1:ny-R; 
 switch fsplitMth
 case 'LF', [fp,fm] = LF(a,q(ic,:,:),[1,0]);
 case 'LLF', [fp,fm] = Rusanov(q(ic,:,:),[1,0]);
 end

 E=4; 
 parfor e=1:E
 switch Recon
 case 'WENO5', [flux(e,:)] = WENO5recon_X(fp(:,:,e),fm(:,:,e),nx);
 case 'WENO7', [flux(e,:)] = WENO7recon_X(fp(:,:,e),fm(:,:,e),nx);
 case 'Poly5', [flux(e,:)] = POLY5recon_X(fp(:,:,e),fm(:,:,e),nx);
 case 'Poly7', [flux(e,:)] = POLY7recon_X(fp(:,:,e),fm(:,:,e),nx);
 end
 end

 res=zeros(size(q)); nc=ny-2*R; nf=nx+1-2*R;
 for e=1:E
 for j=1:nc
 res(j+R,R+1,e) = res(j+R,R+1,e) - flux(e,j+nc*(1-1))/dx;
 for i = 2:nf-1
 res(j+R,i+R-1,e) = res(j+R,i+R-1,e) + flux(e,j+nc*(i-1))/dx;
 res(j+R, i+R ,e) = res(j+R, i+R ,e) - flux(e,j+nc*(i-1))/dx;
 end
 res(j+R,nx-R,e) = res(j+R,nx-R,e) + flux(e,j+nc*(nf-1))/dx;
 end
 end
 clear flux fp fm;
 
 ic=R+1:nx-R;
 switch fsplitMth
 case 'LF', [fp,fm] = LF(a,q(:,ic,:),[0,1]);
 case 'LLF', [fp,fm] = Rusanov(q(:,ic,:),[0,1]);
 end

 parfor e=1:E
 switch Recon
 case 'WENO5', [flux(e,:)] = WENO5recon_Y(fp(:,:,e),fm(:,:,e),ny);
 case 'WENO7', [flux(e,:)] = WENO7recon_Y(fp(:,:,e),fm(:,:,e),ny);
 case 'Poly5', [flux(e,:)] = POLY5recon_Y(fp(:,:,e),fm(:,:,e),ny);
 case 'Poly7', [flux(e,:)] = POLY7recon_Y(fp(:,:,e),fm(:,:,e),ny);
 end
 end

 nc=nx-2*R; nf=ny+1-2*R;
 for e=1:E
 for i=1:nc
 res(R+1,i+R,e) = res(R+1,i+R,e) - flux(e,1+nf*(i-1))/dy;
 for j=2:nf-1
 res(j+R-1,i+R,e) = res(j+R-1,i+R,e) + flux(e,j+nf*(i-1))/dy;
 res( j+R ,i+R,e) = res( j+R ,i+R,e) - flux(e,j+nf*(i-1))/dy;
 end
 res(ny-R,i+R,e) = res(ny-R,i+R,e) + flux(e,nf+nf*(i-1))/dy;
 end
 end
end

function [Fp,Fm] = LF(a,q,normal)
 global gamma
 nx = normal(1); ny = normal(2);
 r=q(:,:,1); u=q(:,:,2)./r; v=q(:,:,3)./r; E=q(:,:,4); vn=u*nx+v*ny;
 p=(gamma-1)*(E-0.5*r.*(u.^2+v.^2));
 F=cat(3, r.*vn, r.*vn.*u + p*nx, r.*vn.*v + p*ny, vn.*(E+p));
 Fp=0.5*(F + a*q); 
 Fm=0.5*(F - a*q); 
end

function [Fp,Fm] = Rusanov(q,normal)
 global gamma
 nx = normal(1); ny = normal(2);
 r = q(:,:,1); u = q(:,:,2)./r; v = q(:,:,3)./r; E = q(:,:,4); vn = u*nx + v*ny;
 p  = (gamma-1)*(E - 0.5*r.*(u.^2 + v.^2));
 a  = sqrt(gamma*p./r); 
 F = cat(3, r.*vn, r.*vn.*u + p*nx, r.*vn.*v + p*ny, vn.*(E + p) );
 alpha = abs(vn) + a; 
 alpha = repmat(alpha,[1 1 4]);
 Fp = 0.5*(F + alpha.*q);
 Fm = 0.5*(F - alpha.*q);
end

function [flux] = WENO5recon_X(v,u,N)
I = 3:(N-3);
vmm = reshape(v(:,I-2,:),1,[]); vm = reshape(v(:,I-1,:),1,[]); vo = reshape(v(:, I ,:),1,[]); vp = reshape(v(:,I+1,:),1,[]); vpp = reshape(v(:,I+2,:),1,[]);
%B0n = 13/12 * (vmm - 2*vm + vo).^2 + 1/4 * (vmm - 4*vm + 3*vo).^2;
%B1n = 13/12 * (vm - 2*vo + vp).^2 + 1/4 * (vm - vp).^2;
%B2n = 13/12 * (vo - 2*vp + vpp).^2 + 1/4 * (3*vo - 4*vp + vpp).^2;
B0n = 1/2*((vm-vmm).^2 + (vo-vm).^2) + (vo-2*vm+vmm).^2;
B1n = 1/2*((vo-vm).^2 + (vp-vo).^2) + (vm-2*vo+vp).^2;
B2n = 1/2*((vp-vo).^2 + (vpp-vp).^2) + (vpp-2*vp+vo).^2;

 b5 = (1/4) * ([1 -4 3] * [vmm; vm; vo]).^2;
 b6 = (1/4) * ([-1 0 1] * [vm; vo; vp]).^2;
 b7 = (1/4) * ([3 -4 1] * [vo; vp; vpp]).^2;
 B5 = (1/144) * ([1 -8 0 8 -1] * [vmm; vm; vo; vp; vpp]).^2;
 
%tau5 = abs(abs(b5-B5)+abs(b7-B5)-2*abs(b6-B5));
tau5 = abs(((vmm - 2*vm + vo).^2) - 2*((vm - 2*vo + vp).^2) + ((vo - 2*vp + vpp).^2));
d0n = 1/10; d1n = 6/10; d2n = 3/10; epsilon = 1e-6; p = 2;
%tau5= abs(B2n-B0n);
alpha0n = d0n * (1 + (tau5 ./ (B0n + epsilon)).^p);
alpha1n = d1n * (1 + (tau5 ./ (B1n + epsilon)).^p);
alpha2n = d2n * (1 + (tau5 ./ (B2n + epsilon)).^p);
%alpha0n = d0n * (1 + (tau5 ./ (B0n + epsilon)).^p);
%alpha1n = d1n * (1 + (tau5 ./ (B1n + epsilon)).^p);
%alpha2n = d2n * (1 + (tau5 ./ (B2n + epsilon)).^p);
%alpha0n = d0n ./(B0n + epsilon).^p;
%alpha1n = d1n ./ (B1n + epsilon).^p;
%alpha2n = d2n ./(B2n + epsilon).^p;
 

alphasumn = alpha0n + alpha1n + alpha2n;
w0n = alpha0n ./ alphasumn; w1n = alpha1n ./ alphasumn; w2n = alpha2n ./ alphasumn;

flux = w0n.*(2*vmm - 7*vm + 11*vo)/6 + w1n.*(-vm + 5*vo + 2*vp)/6 + w2n.*(2*vo + 5*vp - vpp)/6;

umm = reshape(u(:,I-1,:),1,[]); um = reshape(u(:, I ,:),1,[]); uo = reshape(u(:,I+1,:),1,[]); up = reshape(u(:,I+2,:),1,[]); upp = reshape(u(:,I+3,:),1,[]);
%B0p = 13/12 * (umm - 2*um + uo).^2 + 1/4 * (umm - 4*um + 3*uo).^2;
%B1p = 13/12 * (um - 2*uo + up).^2 + 1/4 * (um - up).^2;
%B2p = 13/12 * (uo - 2*up + upp).^2 + 1/4 * (3*uo - 4*up + upp).^2;
B0p = 1/2*((um-umm).^2 + (uo-um).^2) + (uo-2*um+umm).^2;
B1p = 1/2*((uo-um).^2 + (up-uo).^2) + (um-2*uo+up).^2;
B2p = 1/2*((up-uo).^2 + (upp-up).^2) + (upp-2*up+uo).^2;

 b5 = (1/4) * ([1 -4 3] * [umm; um; uo]).^2;
 b6 = (1/4) * ([-1 0 1] * [um; uo; up]).^2;
 b7 = (1/4) * ([3 -4 1] * [uo; up; upp]).^2;
 B5 = (1/144) * ([1 -8 0 8 -1] * [umm; um; uo; up; upp]).^2;
 
%tau5 = abs(abs(b5-B5)+abs(b7-B5)-2*abs(b6-B5));
tau5 = abs(((umm - 2*um + uo).^2) - 2*((um - 2*uo + up).^2) + ((uo - 2*up + upp).^2));
d0p = 1/10; d1p = 6/10; d2p = 3/10; epsilon = 1e-6; p = 2;
%tau5= abs(B2p-B0p);
alpha0p = d0p * (1 + (tau5 ./ (B0p + epsilon)).^p);
alpha1p = d1p * (1 + (tau5 ./ (B1p + epsilon)).^p);
alpha2p = d2p * (1 + (tau5 ./ (B2p + epsilon)).^p);
%alpha0p = d0p * (1 + 0.1.*(tau5 ./ (B0p + epsilon)).^p);
%alpha1p = d1p * (1 + 0.1.*(tau5 ./ (B1p + epsilon)).^p);
%alpha2p = d2p * (1 + 0.8.*(tau5 ./ (B2p + epsilon)).^p);
%alpha0p = d0p ./(B0p + epsilon).^p;
%alpha1p = d1p ./ (B1p + epsilon).^p;
%alpha2p = d2p ./(B2p + epsilon).^p;
 

alphasump = alpha0p + alpha1p + alpha2p;
w0p = alpha0p ./ alphasump; w1p = alpha1p ./ alphasump; w2p = alpha2p ./ alphasump;

flux = flux + w0p.*(-umm + 5*um + 2*uo)/6 + w1p.*(2*um + 5*uo - up)/6 + w2p.*(11*uo - 7*up + 2*upp)/6;
end

function [flux] = WENO5recon_Y(v, u, N)
I = 3:(N-3);
vmm = reshape(v(I-2,:,:),1,[]); vm = reshape(v(I-1,:,:),1,[]); vo = reshape(v(I,:,:),1,[]); vp = reshape(v(I+1,:,:),1,[]); vpp = reshape(v(I+2,:,:),1,[]);
%B0n = 13/12 * (vmm - 2*vm + vo).^2 + 1/4 * (vmm - 4*vm + 3*vo).^2;
%B1n = 13/12 * (vm - 2*vo + vp).^2 + 1/4 * (vm - vp).^2;
%B2n = 13/12 * (vo - 2*vp + vpp).^2 + 1/4 * (3*vo - 4*vp + vpp).^2;
B0n = 1/2*((vm-vmm).^2 + (vo-vm).^2) + (vo-2*vm+vmm).^2;
B1n = 1/2*((vo-vm).^2 + (vp-vo).^2) + (vm-2*vo+vp).^2;
B2n = 1/2*((vp-vo).^2 + (vpp-vp).^2) + (vpp-2*vp+vo).^2;

 b5 = (1/4) * ([1 -4 3] * [vmm; vm; vo]).^2;
 b6 = (1/4) * ([-1 0 1] * [vm; vo; vp]).^2;
 b7 = (1/4) * ([3 -4 1] * [vo; vp; vpp]).^2;
 B5 = (1/144) * ([1 -8 0 8 -1] * [vmm; vm; vo; vp; vpp]).^2;
 
 d0n = 1/10; d1n = 6/10; d2n = 3/10; epsilon = 1e-6;p=2;
 %tau5 = abs(abs(b5-B5)+abs(b7-B5)-2*abs(b6-B5));
 tau5 = abs(((vmm - 2*vm + vo).^2) - 2*((vm - 2*vo + vp).^2) + ((vo - 2*vp + vpp).^2));
 %tau5= abs(B2n-B0n);
 alpha0n = d0n .* (1 +  (tau5 ./ (epsilon + B0n)).^2);
 alpha1n = d1n .* (1 +  (tau5 ./ (epsilon + B1n)).^2);
 alpha2n = d2n .* (1 +  (tau5 ./ (epsilon + B2n)).^2);
 %alpha0n = d0n .* (1 + 0.1.*(tau5 ./ (epsilon + B0n)).^2);
 %alpha1n = d1n .* (1 + 0.1.*(tau5 ./ (epsilon + B1n)).^2);
 %alpha2n = d2n .* (1 + 0.1.*(tau5 ./ (epsilon + B2n)).^2);
 %alpha0n = d0n ./(B0n + epsilon).^p;
 %alpha1n = d1n ./ (B1n + epsilon).^p;
 %alpha2n = d2n./(B2n + epsilon).^p;
 
 alphasumn = alpha0n + alpha1n + alpha2n;
 w0n = alpha0n ./ alphasumn; w1n = alpha1n ./ alphasumn; w2n = alpha2n ./ alphasumn;
 
 flux = w0n .* (2*vmm - 7*vm + 11*vo)/6 + w1n .* (-vm + 5*vo + 2*vp)/6 + w2n .* (2*vo + 5*vp - vpp)/6;

 umm = reshape(u(I-1,:,:),1,[]); um = reshape(u(I,:,:),1,[]); uo = reshape(u(I+1,:,:),1,[]); up = reshape(u(I+2,:,:),1,[]); upp = reshape(u(I+3,:,:),1,[]);
% B0p = 13/12 * (umm - 2*um + uo).^2 + 1/4 * (umm - 4*um + 3*uo).^2;
% B1p = 13/12 * (um - 2*uo + up).^2 + 1/4 * (um - up).^2;
% B2p = 13/12 * (uo - 2*up + upp).^2 + 1/4 * (3*uo - 4*up + upp).^2;
 B0p = 1/2*((um-umm).^2 + (uo-um).^2) + (uo-2*um+umm).^2;
 B1p = 1/2*((uo-um).^2 + (up-uo).^2) + (um-2*uo+up).^2;
 B2p = 1/2*((up-uo).^2 + (upp-up).^2) + (upp-2*up+uo).^2;

 b5 = (1/4) * ([1 -4 3] * [umm; um; uo]).^2;
 b6 = (1/4) * ([-1 0 1] * [um; uo; up]).^2;
 b7 = (1/4) * ([3 -4 1] * [uo; up; upp]).^2;
 B5 = (1/144) * ([1 -8 0 8 -1] * [umm; um; uo; up; upp]).^2;
 
 d0p = 1/10; d1p = 6/10; d2p = 3/10; epsilon = 1e-6; p = 2;
 %tau5 = abs(abs(b5-B5)+abs(b7-B5)-2*abs(b6-B5));
 tau5 = abs(((umm - 2*um + uo).^2) - 2*((um - 2*uo + up).^2) + ((uo - 2*up + upp).^2));
 %tau5= abs(B2p-B0p);
 alpha0p = d0p .* (1 + (tau5 ./ (epsilon + B0p)).^2);
 alpha1p = d1p .* (1 + (tau5 ./ (epsilon + B1p)).^2);
 alpha2p = d2p .* (1 + (tau5 ./ (epsilon + B2p)).^2);
 %alpha0p = d0p .* (1 + 0.1.*(tau5 ./ (epsilon + B0p)).^2);
 %alpha1p = d1p .* (1 + 0.1.*(tau5 ./ (epsilon + B1p)).^2);
 %alpha2p = d2p .* (1 + 0.8.*(tau5 ./ (epsilon + B2p)).^2);
 % Nonlinear alpha weights (WENO-Z)
 %alpha0p = d0p ./(B0p + epsilon).^p;
 %alpha1p = d1p ./ (B1p + epsilon).^p;
 %alpha2p = d2p ./(B2p + epsilon).^p;
 
 alphasump = alpha0p + alpha1p + alpha2p;
 w0p = alpha0p ./ alphasump; w1p = alpha1p ./ alphasump; w2p = alpha2p ./ alphasump;
 
 flux = flux + w0p .* (-umm + 5*um + 2*uo)/6 + w1p .* (2*um + 5*uo - up)/6 + w2p .* (11*uo - 7*up + 2*upp)/6;
end
