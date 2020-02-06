function [M_matrix,U_matrix] = VirtualLMNN(data,label,Beta,Mu)
[~,n_fea] = size(data); % sample num, feature num
M_matrix = eye(n_fea,n_fea);           % orign B,metric matrix
L_matrix = eye(n_fea,n_fea);
EIG_matrix = eye(n_fea,n_fea);
U_matrix = data; % origin

[U_matrix] = compute_Ui(data,label,U_matrix,M_matrix,L_matrix,EIG_matrix,Beta,Mu);
[M_matrix,~,~] = my_lmnn(data,U_matrix,M_matrix,label,Beta);
end
function [U_matrix_new] = compute_Ui(Data,label,U_matrix,M_matrix,L_matrix,EIG_matrix,Beta,Mu)
% compute the virtual matrix n*d ¼ÆËãÐéÄâµã¾ØÕó
% Data:              a n*d origin dataset
% label:             a n*1 label column vector 
% U_matrix:          a n*d virtual dataset
% M_matrix:          a d*d metric matrix
% L_matrix£º         a feature vector matrix of M_matrix 
% EIG_matrix£º       a Eigenvalues matrix of M_matrix
% Beta,Mu :      regularization parameter
% return U_matrix_new 
    [n_sam,n_fea] = size(Data); 
    U_matrix_new = zeros(n_sam,n_fea);
    for i = 1:n_sam
        Pl_index = find(label(i) ~= label); % xi and xl unsimilar label        
        Ck_index = find(label(i) == label); % C_k: xi class
        nCk = length(Ck_index);                 % |C_k| length
        x_i = Data(i,:);
        u_i = U_matrix(i,:);
        dist_ii = (u_i - x_i) * M_matrix * (u_i - x_i)';
        
        G_index = [];
        for j = 1:nCk
            if U_matrix(Ck_index(j),:) ~= u_i
                G_index = [G_index;Ck_index(j)];
            end
        end
        nG = size(G_index,1);
 
        EIG_flag = zeros(n_fea,n_fea);
        diag_index = sub2ind(size(EIG_matrix),1:n_fea,1:n_fea);
        EIG_flag(diag_index) = 1./(EIG_matrix(diag_index) + Beta * nG * ones(1,n_fea));
        part_left = L_matrix * EIG_flag * L_matrix';
        
        part_right1 = M_matrix * x_i';
        u_j_avg = zeros(n_fea,1);
        for j = 1:nG
            u_j = U_matrix(G_index(j),:);
            u_j_avg = u_j_avg + u_j';
        end
        part_right2 = u_j_avg .* Beta;
        
        part_right3 = zeros(n_fea,1);
        for m = 1: length(Pl_index)
            x_m =  Data(Pl_index(m),:);
            dist_im = (u_i - x_m) * M_matrix * (u_i - x_m)';
            if 1 + dist_ii >= dist_im
                part_right3 = part_right3 + Mu * M_matrix * (x_m - x_i)';
            end
        end
        newu_i = part_left * (part_right1 + part_right2 - part_right3);
        U_matrix_new(i,:) = newu_i';
    end
end




