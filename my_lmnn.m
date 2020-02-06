function [M, L, S] = my_lmnn(Data,U_matrix,M_matrix,labels,REGUL_BELTA)

%   [M, L, Y, C] = lmnn(X, labels)


    % Initialize some variables
    [N, D] = size(Data);
    assert(length(labels) == N);
    C = Inf; prev_C = Inf;
    
    % Set learning parameters
    min_iter = 50;          % minimum number of iterations
    max_iter = 1000;        % maximum number of iterations
    eta = 1e-1;               % learning rate
    tol = 1e-3;             % tolerance for convergence
    best_C = Inf;           % best error obtained so far
    best_M = M_matrix;             % best metric found so far
  
    % Compute pulling term between target neigbhors to initialize gradient
    slack = zeros(N, N);        
    G = zeros(D, D);
    for i=1:N
        G = G +  (Data(i,:)-U_matrix(i,:))' * (Data(i,:)-U_matrix(i,:));
    end
    % Perform main learning iterations
    iter = 0;
    while (prev_C - C > tol || iter < min_iter) && iter < max_iter
        Dist_ii_vec = zeros(N,1);
        for i=1:N
            Dist_ii_vec(i) = (U_matrix(i,:)-Data(i,:)) * M_matrix * (U_matrix(i,:)-Data(i,:))';
        end
        
        Dist_ij_matrix = zeros(N,N);
        for i = 1:N
             unsim_index = find(labels ~=labels(i));
             n_unsim = length(unsim_index);
             for j = 1: n_unsim
                Dist_ij_matrix(i,unsim_index(j)) = (U_matrix(i,:)-Data(unsim_index(j),:)) * M_matrix * (U_matrix(i,:)-Data(unsim_index(j),:))';
             end
        end
    
        % Compute value of slack variables
        old_slack = slack;
        for i=1:N
            unsim_index = find(labels ~=labels(i));
            n_unsim = length(unsim_index);
            dist_ii = Dist_ii_vec(i);
            for j = 1:n_unsim
                dist_ij = Dist_ij_matrix(i,unsim_index(j));
                slack(i,unsim_index(j)) = max(1+ dist_ii - dist_ij,0);
            end
        end
        
        % Compute value of cost function
        prev_C = C;
        C = sum(Dist_ii_vec) + sum(slack(:));
        
        % Maintain best solution found so far (subgradient method)
        if C < best_C
            best_C = C;
            best_M = M_matrix;
        end
        
        % Perform gradient update
        
        [r, c] = find(slack(:,:) > 0 & old_slack(:,:) == 0);
        for i = 1:length(r)
            G_ii = (U_matrix(r(i),:) - Data(r(i),:))' * (U_matrix(r(i),:) - Data(r(i),:));
            G_ij = (U_matrix(r(i),:) - Data(c(i),:))' * (U_matrix(r(i),:) - Data(c(i),:));
            G = G + REGUL_BELTA*(G_ii-G_ij);
        end            
        % Remove terms for resolved violations
        [r, c] = find(slack(:,:) == 0 & old_slack(:,:) > 0);
        for i = 1:length(r)
            G_ii = (U_matrix(r(i),:) - Data(r(i),:))' * (U_matrix(r(i),:) - Data(r(i),:));
            G_ij = (U_matrix(r(i),:) - Data(c(i),:))' * (U_matrix(r(i),:) - Data(c(i),:));
            G = G - REGUL_BELTA* (G_ii-G_ij);
        end
        M_matrix = M_matrix - (eta ./ N) .* G;
        
        % Project metric back onto the PSD cone
%         M_matrix = (M_matrix + M_matrix')/2;
        [V, L] = eig(M_matrix);
        V = real(V); L = real(L);
        ind = find(diag(L) > 0);
        if isempty(ind)
            warning('Projection onto PSD cone failed. All eigenvalues were negative.'); break
        end
        M_matrix = V(:,ind) * L(ind, ind) * V(:,ind)';
        if any(isinf(M_matrix(:)))
            warning('Projection onto PSD cone failed. Metric contains Inf values.'); break
        end
        if any(isnan(M_matrix(:)))
            warning('Projection onto PSD cone failed. Metric contains NaN values.'); break
        end
        
        % Update learning rate
        if prev_C > C
            eta = eta * 1.01;
        else
            eta = eta * .5;
        end
        
        % Print out progress
        iter = iter + 1;
        no_slack = sum(slack(:) > 0);
        if rem(iter, 10) == 0
            disp(['Iteration ' num2str(iter) ': error is ' num2str(C ./ N) ...
                  ', number of constraints: ' num2str(no_slack)]);
        end
    end
    
    % Return best metric and error
    M = best_M;
    C = best_C;
    
    % Compute mapped data
    [L, S, ~] = svd(M);
    L = bsxfun(@times, sqrt(diag(S)), L);
end

function x = vec(x)
    x = x(:);
end
