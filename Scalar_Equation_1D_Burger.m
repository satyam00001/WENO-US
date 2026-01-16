function Scalar_Equation_1D_Burger
    Study = 1; % 0 = Linear Wave, 1 = Burgers
    x0 = -1; % Left Boundary
    x1 = 1; % Right Boundary
    IC = 2 ; % Initial condition type
    %k = 1; % Parameter
    N = 2000; % Number of Grid points
    cfl = 0.45; % CFL number 
    %Final_Time = 1 ; % Final time for case 1
    Final_Time = 0.5 ; % Final time 
    Order = 5; % WENO Order
    epsilon = 10^(-12); % WENO sensitivity parameter
    p = 2; % WENO power parameter
    r = (Order + 1) / 2;
    dx = abs(x1 - x0) / N; % Grid Spacing
    x = x0 + (0:N-1) * dx; % Grid space
    switch Study
        case 0
            Flux = @Flux_Linear_Wave; Flux_Jacobian = @Jacobian_Linear_Wave;
        case 1
            Flux = @Flux_Burgers; Flux_Jacobian = @Jacobian_Burgers;
    end
    
    Step = 0; Time = 0; Q = Initial_Condition(IC); % Initial Condition
    
    % Start measuring CPU time
    tic;
    while Time < Final_Time
        dt = cfl * dx;
        if dt > (Final_Time - Time), dt = Final_Time - Time; end
        
        % 3rd order TVD Runge-Kutta scheme
        LQ = -Diff(Flux(BoundaryCondition(Q)), BoundaryCondition(Q));
        Q1 = Q + dt * LQ;
        LQ = -Diff(Flux(BoundaryCondition(Q1)), BoundaryCondition(Q1));
        Q2 = (3*Q + Q1 + dt * LQ) / 4;
        LQ = -Diff(Flux(BoundaryCondition(Q2)), BoundaryCondition(Q2));
        Q = (Q + 2*Q2 + 2*dt * LQ) / 3;
        
        Step = Step + 1; Time = Time + dt;
        Output(Q);
        pause(0.1);
    end  
    
    % Stop measuring CPU time
    elapsed_time = toc;
    % Display elapsed CPU time
    disp(['CPU time: ', num2str(elapsed_time), ' seconds']);
    
    %------------------------Functions-----------------------
    function Q = Initial_Condition(IC)
        switch IC % function for IC
            case 0
                Q = -sin(pi * x);
            case 1
                Q =  0.5 + 0.5* sin(pi * x) ;
            case 2
                Q = zeros(size(x));
                Q(x >= -0.5 & x <= 0.5) = 1; % u(x,0) = 1 for x < 0
                %Q(x) = 0; % u(x,0) = 0 for x >= 0
        end
    end
    
    % Linear Wave Equation
    function F = Flux_Linear_Wave(Q)
        F = a * Q;
    end
    function dFdQ = Jacobian_Linear_Wave(~)
        dFdQ = a;
    end
    
    % Burgers Equation
    function F = Flux_Burgers(Q)
        F = Q .* Q / 2;
    end
    function dFdQ = Jacobian_Burgers(Q)
        dFdQ = Q;
    end
    
    %------------------------Boundary Condition----------------------
    function f_BC = BoundaryCondition(f)
        f_BC = [f(end-2:end), f, f(1:3)]; % Periodical BC
    end

    %------------------------WENO Differentiation--------------------
    function D_F = Diff(F, Q)
        r = (Order + 1) / 2; NN = length(F); D_F = zeros(1, NN - 2 * r);
        
        % Global Lax-Friedrichs flux splitting at cell center
        alpha = max(abs(Flux_Jacobian(Q)));
        F_plus = (F + alpha * Q) / 2;
        F_minus = (F - alpha * Q) / 2;
        
        % WENO Reconstruction at cell boundary
        F_half_plus = WENO_Reconstruction(F_plus);
        G_minus = F_minus(end:-1:1);
        G_minus = WENO_Reconstruction(G_minus);
        F_half_minus = G_minus(end:-1:1);
        
        F_half = F_half_plus + F_half_minus;
        
        % Differentiation of the flux f(x) at cell center
        for i = r + 1:NN - r
            F_half(i+1) = F_half(i+1) ...
            - (1/24)*dx^2*second_derivative(F_half(:,i), dx) ...
             + (7/5760)*dx^4*fourth_derivative(F_half(:,i), dx);

            D_F(i - r) = (F_half(i + 1) - F_half(i)) / dx;
        end
    end
    
   %------------------------WENO Reconstruction---------------------
function f_half = WENO_Reconstruction(f)
    r = (Order+1)/2 ; NN = length(f); f_half = zeros(1,NN+1);
    %d = [1/10, 6/10, 3/10];
    d = [1/16, 5/8, 5/16];
    c1 = 1/4 ; c2 = 13/12 ;% Ideal Weights
    s2 = 1/12 ;
    s1 = 1/4;
    f = f(:) ; % Makes f a column vector
    P = zeros(3,1); % P is a column vector
    beta = zeros(1,3);
    R = zeros(1,3);
    S = zeros(1,3);
    
    for i = r:NN-r
   % lower order polynomial in substencil S^k, k=0,1,2 using cell center value f_i
            %P(1) = [ 2 -7 11]/6 * f(i-2:i );
            %P(2) = [-1 5 2]/6 * f(i-1:i+1);
            %P(3) = [ 2 5 -1]/6 * f( i:i+2);
            P(1) = [ 3 -10 15]/8 * f(i-2:i );
            P(2) = [-1 6 3]/8 * f(i-1:i+1);
            P(3) = [ 3 6 -1]/8 * f( i:i+2);
    % local lower order smoothness indicator in substencil S^k, k=0,1,2
        beta(1) = c1*([ 1 -4 3] * f(i-2:i )).^2 + c2*([1 -2 1] * f(i-2:i )).^2;
        beta(2) = c1*([-1 0 1] * f(i-1:i+1)).^2 + c2*([1 -2 1] * f(i-1:i+1)).^2;
        beta(3) = c1*([-3 4 -1] * f( i:i+2)).^2 + c2*([1 -2 1] * f( i:i+2)).^2;
        %beta(1) = 1/2*(([ -1 1 0] * f(i-2:i )).^2 + ([0 -1 1] * f(i-2:i )).^2) + ([1 -2 1] * f(i-2:i)).^2;
        %beta(2) = 1/2*(([ -1 1 0] * f(i-1:i+1 )).^2 + ([0 -1 1] * f(i-1:i+1)).^2) + ([1 -2 1] * f(i-1:i+1)).^2;
        %beta(3) = 1/2*(([ -1 1 0] * f(i:i+2 )).^2 + ([0 -1 1] * f(i:i+2 )).^2) + ([1 -2 1] * f(i:i+2)).^2;
        S(1) = s1*([ 1 -4 3] * f(i-2:i )).^2 + s2*([1 -2 1] * f(i-2:i )).^2;
        S(2) = s1*([-1 0 1] * f(i-1:i+1)).^2 + s2*([1 -2 1] * f(i-1:i+1)).^2;
        S(3) = s1*([-3 4 -1] * f( i:i+2)).^2 + s2*([1 -2 1] * f( i:i+2)).^2;
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
        %tau = abs((3*B(5)+b(5))/4 - (3*b(2)+6*b(3)-b(4))/8);
        %tau = abs(b(1)-3*b(2)+3*b(3)-b(4))/6 ;
        % WENO-JS weights : alpha_k
         %alpha = d./(beta + epsilon).^p;
        %WENO-Z weights : alpha_k
        %tau = abs(beta(3) - beta(1));
        %E = [1/10,1/10,4/5];
        %E = [3/4,3/2,3/4] ;
        %D = 20;
        %tau = abs(-2*([1 -2 1] * f(i-1:i+1)).^2 +  ([1 -2 1] * f(i-2:i)).^2 +([1 -2 1] * f(i:i+2)).^2);
        tau= (abs(b(6)-B(5)) + abs(b(7)-B(5)) -2* abs(b(5)-B(5)));
        %alpha = d.*( 1 + E.*(tau./( beta + epsilon) ).^p );
        %alpha = d.*( 1 +( tau./( beta + epsilon) ).^p );
        omega = alpha/sum(alpha);
       % Reconstructed polynomial f_half at the cell boundary x_(i+1/2)
        f_half(i+1) = omega*P;
    end
end
    
    function Output(Q)
        % Calculate the exact solution at the final time
        %u_exact = exact_solution(x, Final_Time);
        % Plot the numerical solution and the exact solution
        plot(x, Q, 'k','MarkerSize', 2);   
        %plot(x, u_exact, 'r--'); hold off;
        legend('Exact');
        xlabel('x');
        ylabel('u');
        %title(['Time = ', num2str(Time)]);
        drawnow;
    end
end


