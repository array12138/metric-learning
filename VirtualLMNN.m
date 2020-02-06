function [M_matrix,U_matrix] = VirtualLMNN(data,label,Beta,Mu)
% compute const
[~,n_fea] = size(data); % sample num, feature num
label_vec = unique(label);
n_K = length(label_vec);

%初始的矩阵向量
M_matrix = eye(n_fea,n_fea);           % orign B,metric matrix
L_matrix = eye(n_fea,n_fea);
EIG_matrix = eye(n_fea,n_fea);
U_matrix = data; % origin

maxcount = 20;
count_all = 1;

while true
    M_matrix_old = M_matrix; 
    [U_matrix] = compute_Ui(data,label,U_matrix,M_matrix,L_matrix,EIG_matrix,Beta,Mu);
    [M_matrix,~,~] = my_lmnn(data,U_matrix,M_matrix,label,Beta);
    M_matrix = (M_matrix+M_matrix')/2;
    [L_matrix,EIG_matrix] = eig(M_matrix);
    iter1 = norm(M_matrix-M_matrix_old,'fro')/max(1,norm(M_matrix_old,'fro'));
    disp(strcat('M_ratio = ',num2str(iter1)));
    if count_all ==maxcount || iter1<= 0.01
        break;
    end
    count_all = count_all + 1;
end 
end
function [U_matrix_new] = compute_Ui(Data,label,U_matrix,M_matrix,L_matrix,EIG_matrix,Beta,Mu)
% compute the virtual matrix n*d 计算虚拟点矩阵
% Data:              a n*d origin dataset
% label:             a n*1 label column vector 
% U_matrix:          a n*d virtual dataset
% M_matrix:          a d*d metric matrix
% L_matrix：         a feature vector matrix of M_matrix 
% EIG_matrix：       a Eigenvalues matrix of M_matrix
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




