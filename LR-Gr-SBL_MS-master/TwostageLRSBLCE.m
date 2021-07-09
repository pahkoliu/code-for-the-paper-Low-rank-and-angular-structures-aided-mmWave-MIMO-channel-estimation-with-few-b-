function [NMSE_MUSIC,NMSE] = TwostageLRSBLCE(Y, X, H, h_norm_est, Real_Imag_y_bin, Num_Iter, sigma, Quantizer)


    Array_fun = @(Nvar,thetavar) exp(1j*(0:1:Nvar-1)'*thetavar');

    a =0;                                     % parameters of the gamma distribution, usually set vey low, e.g., 10^-4 or 0 for SBL algorithm
    b = 0;
    [m,t] = size(Y);
    [~,n] = size(H);
    r_init = 10;
    r_init = min([r_init,round(min([m,n])/2)]);         
    dampfac =0.4;
    DIMRED = 1;
    DIMRED_THR = 1e2;
    tol = 1e-4;
    % From A to B
    h_A_ext_mat = 0*ones(m,n);
    v_h_A_ext_mat = 1e1*ones(m,n);
    % From B to C
    z_B_ext_mat = 0*ones(m,t);
    v_z_B_ext_mat = 1e1*ones(m,t);
    Bit = Quantizer.Bit;
    wvar_hat = sigma;
%     if Bit<inf
%         wvar_hat = sigma;
%     else
%         wvar_hat = 1;
%     end
    
    %% LR-SBL-CE Algorithm Implemention
    if Bit<inf
        v_z_B_ext = v_z_B_ext_mat(:);
        z_B_ext = z_B_ext_mat(:);
        z_B_ext_real = [real(z_B_ext);imag(z_B_ext)];
        v_B_ext_real = [v_z_B_ext;v_z_B_ext]/2;
        [Real_Imag_z_C_post, Real_Imag_v_z_C_post] = MMSE(Real_Imag_y_bin, z_B_ext_real, v_B_ext_real, Quantizer, wvar_hat/2);
        z_C_post = Real_Imag_z_C_post(1:m*t)+1j*Real_Imag_z_C_post(m*t+1:end);
        v_z_C_post = Real_Imag_v_z_C_post(1:m*t)+Real_Imag_v_z_C_post(m*t+1:end);
        z_C_post_mat = reshape(z_C_post,m,t);
        v_z_C_post_mat = reshape(v_z_C_post,m,t);
        v_z_C_ext_mat = v_z_C_post_mat.*v_z_B_ext_mat./(v_z_B_ext_mat-v_z_C_post_mat+eps);
        z_C_ext_mat = v_z_C_ext_mat.*(z_C_post_mat./(v_z_C_post_mat)-z_B_ext_mat./(v_z_B_ext_mat));
    else
        v_z_C_ext_mat = wvar_hat*ones(m,t);
        z_C_ext_mat = Y;
    end

    % From B to A, transform to row processing
    h_B_post_mat_trans = nan(n,m); 
    v_h_B_post_mat_trans = nan(n,m); 
    for row_ind = 1:m
        h_A_ext_mat_row = h_A_ext_mat(row_ind,:).';
        v_h_A_ext_mat_row = v_h_A_ext_mat(row_ind,:).';
        z_tilde_row = z_C_ext_mat(row_ind,:).';
        v_z_C_row = v_z_C_ext_mat(row_ind,:).';

        Sigma_h_B_post = inv(X'*diag(1./v_z_C_row)*X+diag(1./v_h_A_ext_mat_row));
        h_B_post_mat_trans(:,row_ind) = Sigma_h_B_post*(X'*(z_tilde_row./v_z_C_row)+(h_A_ext_mat_row./v_h_A_ext_mat_row));
        v_h_B_post_mat_trans(:,row_ind) = diag(Sigma_h_B_post);
    end  
    h_B_post_mat = h_B_post_mat_trans.';
    v_h_B_post_mat = v_h_B_post_mat_trans.';

    v_h_B_ext_mat = v_h_A_ext_mat.*v_h_B_post_mat./(v_h_A_ext_mat-v_h_B_post_mat);
    h_B_ext_mat = v_h_B_ext_mat.*(h_B_post_mat./v_h_B_post_mat-h_A_ext_mat./v_h_A_ext_mat);

    Cov_H = nan(m,n);

    Y_tilde = h_B_ext_mat;
    Y2sum = sum(abs(Y_tilde(:)).^2);
    scale2 = Y2sum / (m*n);  % variance of Y
    scale = sqrt(scale2)/10;


    [U, S, V] = svd(Y_tilde, 'econ');
    A = U(:,1:r_init)*(S(1:r_init,1:r_init)).^(0.5);
    B = (S(1:r_init,1:r_init)).^(0.5)*V(:,1:r_init)';
    B = B';

    Sigma_A = repmat( scale*eye(r_init,r_init), [1 1 m] );          %一共m行，每一行的协方差矩阵
    Sigma_B = repmat( scale*eye(r_init,r_init), [1 1 n] );

    gammas = (m + n+ a )./( diag(B'*B) + diag(sum(Sigma_B,3)) + diag(A'*A)+ diag(sum(Sigma_A,3)) + b); 
    gammas = real(gammas);
    v_h_B_ext_mat_inv = 1./v_h_B_ext_mat;



    Hhat = A*B';

    NMSE = nan(Num_Iter,1);
    if Bit==1
        Hhat = Hhat/norm(Hhat,'fro')*h_norm_est;
    end
    NMSE(1) = 20*log10(norm(Hhat-H,'fro')/norm(H,'fro'));
    % if Bit>1
    %     NMSE(1) = 20*log10(norm(Hhat-H,'fro')/norm(H,'fro'));
    % else
    %     NMSE(1) = 20*log10(norm(Hhat/norm(Hhat,'fro')*h_norm_est-H,'fro')/norm(H,'fro'));
    % %     H_hat_vec = Hhat(:);
    % %     deb_c = Hhat(:)'*H(:)/(Hhat(:)'*Hhat(:)+eps);
    % %     MSE_outer(1) = 20*log10(norm(deb_c*Hhat-H_true,'fro')/norm(H_true,'fro'));
    % end


    for Iter = 1:Num_Iter-1
%         waitbar(Iter/Num_Iter)
        Aw_inv = diag(gammas);  
        betaY = v_h_B_ext_mat_inv.*h_B_ext_mat;
         % A step
         for i=1:m                      %iterate over rows
            Bibeta = bsxfun(@times,B,sqrt(v_h_B_ext_mat_inv(i,:)'));
            Sigma_Bbeta = bsxfun(@times,Sigma_B(:,:,:),reshape(v_h_B_ext_mat_inv(i,:),1,1,[]));
            Sigma_A(:,:,i) = (Bibeta'*Bibeta + sum(Sigma_Bbeta,3) + Aw_inv)^(-1);
            A(i,:) = betaY(i,:)*B*Sigma_A(:,:,i);
        end

        % B step
        for j=1:n %Iterate over cols
            Ajbeta = bsxfun(@times,A,sqrt(v_h_B_ext_mat_inv(:,j)));
            Sigma_Abeta = bsxfun(@times,Sigma_A(:,:,:),reshape(v_h_B_ext_mat_inv(:,j),1,1,[]));
            Sigma_B(:,:,j) = (Ajbeta'*Ajbeta + sum(Sigma_Abeta,3) + Aw_inv)^(-1);
            B(j,:) = betaY(:,j)'*A*Sigma_B(:,:,j);
        end

        % estimate gammas
       gammas = (m + n +a)./( diag(B'*B) + diag(sum(Sigma_B,3)) + diag(A'*A)+ diag(sum(Sigma_A,3)) +b);  
       gammas = real(gammas);        


         % Prune irrelevant dimensions?   
        if DIMRED
            MAX_GAMMA = min(gammas) * DIMRED_THR;       
            if sum(find(gammas > MAX_GAMMA))        
                index = find(gammas <= MAX_GAMMA);            
                A = A(:,index);
                B = B(:,index);
                gammas = gammas(index);           
                Sigma_A = Sigma_A(index,index,:);
                Sigma_B = Sigma_B(index,index,:);
            end 
        end      



        % Update the posterior means and variances of H
        Hhat = A*B';

        for ki = 1:m
            Sigma_A_ki = Sigma_A(:,:,ki);
            Sigma_A_ki_t = Sigma_A_ki';
            A_ki_mtx = A(ki,:)'*A(ki,:);
            A_ki_mtx_t = A_ki_mtx';
            for kj = 1:n
                B_kj_mtx = B(kj,:)'*B(kj,:);
                B_kj_mtx_t = B_kj_mtx';
                Sigma_B_kj = Sigma_B(:,:,kj);
                Cov_H(ki,kj) = B_kj_mtx_t(:)'*Sigma_A_ki_t(:) + A_ki_mtx_t(:)'*Sigma_B_kj(:)...
                    + Sigma_A_ki_t(:)'*Sigma_B_kj(:);
            end
        end            
         Cov_H = (Cov_H+conj(Cov_H))/2;

        v_h_A_ext_mat = real(Cov_H).*v_h_B_ext_mat./(v_h_B_ext_mat-real(Cov_H)+eps);
        h_A_ext_mat = v_h_A_ext_mat.*(Hhat./Cov_H-h_B_ext_mat./v_h_B_ext_mat);
%         if Iter>1
%              v_h_A_ext_mat = (1-dampfac)*v_h_A_ext_mat_old+dampfac*v_h_A_ext_mat;
%              h_A_ext_mat = (1-dampfac)*h_A_ext_mat_old+dampfac*h_A_ext_mat;
%          end
%          h_A_ext_mat_old = h_A_ext_mat;
%          v_h_A_ext_mat_old = v_h_A_ext_mat;

        % transform to row processing
        h_B_post_mat_trans = nan(n,m); 
        z_B_post_mat_trans = nan(t,m); 
        v_z_B_post_mat_trans = nan(t,m); 
        for row_ind = 1:m
            h_A_ext_mat_row = h_A_ext_mat(row_ind,:).';
            v_h_A_ext_mat_row = v_h_A_ext_mat(row_ind,:).';
            z_tilde_row = z_C_ext_mat(row_ind,:).';
            v_z_C_row = v_z_C_ext_mat(row_ind,:).';

            Sigma_h_B_post = inv(X'*diag(1./v_z_C_row)*X+diag(1./v_h_A_ext_mat_row));
            h_B_post_mat_trans(:,row_ind) = Sigma_h_B_post*(X'*(z_tilde_row./v_z_C_row)+(h_A_ext_mat_row./v_h_A_ext_mat_row));
            z_B_post_mat_trans(:,row_ind) = X*h_B_post_mat_trans(:,row_ind);
            v_z_B_post_mat_trans(:,row_ind) = diag(X*Sigma_h_B_post*X');
        end

        z_B_post_mat = z_B_post_mat_trans.';
        v_z_B_post_mat = real(v_z_B_post_mat_trans.');




        v_z_B_ext_mat = v_z_C_ext_mat.*v_z_B_post_mat./(v_z_C_ext_mat-v_z_B_post_mat);
        z_B_ext_mat = v_z_B_ext_mat.*(z_B_post_mat./v_z_B_post_mat-z_C_ext_mat./v_z_C_ext_mat);

         if Iter>1
             v_z_B_ext_mat = (1-dampfac)*v_z_B_ext_mat_old+dampfac*v_z_B_ext_mat;
             z_B_ext_mat = (1-dampfac)*z_B_ext_mat_old+dampfac*z_B_ext_mat;
         end
         z_B_ext_mat_old = z_B_ext_mat;
         v_z_B_ext_mat_old = v_z_B_ext_mat;

        Zhat = z_B_post_mat;
%         if Bit==inf
%     %         wvar_hat = (norm((Hhat*X.'-z_C_ext_mat),'fro')^2+sum(sum(v_z_B_post_mat)))/m/n;          
%             wvar_hat = (norm((Zhat-z_C_ext_mat),'fro')^2+sum(sum(v_z_B_post_mat)))/m/n;          
%         end

        if Bit<inf
            v_z_B_ext = v_z_B_ext_mat(:);
            z_B_ext = z_B_ext_mat(:);
            z_B_ext_real = [real(z_B_ext);imag(z_B_ext)];
            v_B_ext_real = [v_z_B_ext;v_z_B_ext]/2;
            [Real_Imag_z_C_post, Real_Imag_v_z_C_post] = MMSE(Real_Imag_y_bin, z_B_ext_real, v_B_ext_real, Quantizer, wvar_hat/2);
            z_C_post = Real_Imag_z_C_post(1:m*t)+1j*Real_Imag_z_C_post(m*t+1:end);
            v_z_C_post = Real_Imag_v_z_C_post(1:m*t)+Real_Imag_v_z_C_post(m*t+1:end);
            z_C_post_mat = reshape(z_C_post,m,t);
            v_z_C_post_mat = reshape(v_z_C_post,m,t);
            v_z_C_ext_mat = v_z_C_post_mat.*v_z_B_ext_mat./(v_z_B_ext_mat-v_z_C_post_mat+eps);
            z_C_ext_mat = v_z_C_ext_mat.*(z_C_post_mat./(v_z_C_post_mat)-z_B_ext_mat./(v_z_B_ext_mat));
        else
            v_z_C_ext_mat = wvar_hat*ones(m,t);
            z_C_ext_mat = Y;
        end

         if Iter>1
             v_z_C_ext_mat = (1-dampfac)*v_z_C_ext_mat_old+dampfac*v_z_C_ext_mat;
             z_C_ext_mat = (1-dampfac)*z_C_ext_mat_old+dampfac*z_C_ext_mat;
         end
         v_z_C_ext_mat_old = v_z_C_ext_mat;
         z_C_ext_mat_old = z_C_ext_mat;

       % transform to row processing
        h_B_post_mat_trans = nan(n,m); 
        v_h_B_post_mat_trans = nan(n,m); 
        for row_ind = 1:m
            h_A_ext_mat_row = h_A_ext_mat(row_ind,:).';
            v_h_A_ext_mat_row = v_h_A_ext_mat(row_ind,:).';
            z_tilde_row = z_C_ext_mat(row_ind,:).';
            v_z_C_row = v_z_C_ext_mat(row_ind,:).';

            Sigma_h_B_post = inv(X'*diag(1./v_z_C_row)*X+diag(1./v_h_A_ext_mat_row));
            h_B_post_mat_trans(:,row_ind) = Sigma_h_B_post*(X'*(z_tilde_row./v_z_C_row)+(h_A_ext_mat_row./v_h_A_ext_mat_row));
            v_h_B_post_mat_trans(:,row_ind) = diag(Sigma_h_B_post);
        end

        h_B_post_mat = h_B_post_mat_trans.';
        v_h_B_post_mat = v_h_B_post_mat_trans.';     





        v_h_B_ext_mat = v_h_A_ext_mat.*v_h_B_post_mat./(v_h_A_ext_mat-v_h_B_post_mat);
        v_h_B_ext_mat = real(v_h_B_ext_mat);
        h_B_ext_mat = v_h_B_ext_mat.*(h_B_post_mat./v_h_B_post_mat-h_A_ext_mat./v_h_A_ext_mat);
        if Iter>1
             h_B_ext_mat = (1-dampfac)*h_B_ext_mat_old+dampfac*h_B_ext_mat;
             v_h_B_ext_mat = (1-dampfac)*v_h_B_ext_mat_old+dampfac*v_h_B_ext_mat;
        end
        h_B_ext_mat_old = h_B_ext_mat;
        v_h_B_ext_mat_old = v_h_B_ext_mat;
        v_h_B_ext_mat_inv = 1./v_h_B_ext_mat;

        
    %           MSE of SBL
        if Bit==1
            Hhat = Hhat/norm(Hhat,'fro')*h_norm_est;
        end
        if Iter>2&&norm(Hhat-Hhat_old,'fro')/norm(Hhat,'fro')<tol
            NMSE(Iter+1:end)  = NMSE(Iter);
            break;
        end
        Hhat_old = Hhat;
        NMSE(Iter+1) = 20*log10(norm(Hhat-H,'fro')/norm(H,'fro'));


    end


    rhat = rank(A);



    %% MUSIC



    R_A = A*A';
    R_B = B*B';
    [W1,~] = rootmusic(R_A,rhat,'corr');


    [W2,~] = rootmusic(R_B,rhat,'corr');



    a_music = repmat([1j*(0:1:m-1)'],1,rhat);
    b_music = repmat([1j*(0:1:n-1)'],1,rhat);

    a_musicW1 = bsxfun(@times,a_music,W1');
    b_musicW2 = bsxfun(@times,b_music,W2');

    a_musicA = exp(a_musicW1);
    a_musicB = exp(b_musicW2);


    R_A_music = zeros(m*m,rhat);
    R_B_music = zeros(n*n,rhat);

    for i = 1:rhat
        R_A_music(:,i) = kron(conj(a_musicA(:,i)),a_musicA(:,i));
        R_B_music(:,i) = kron(conj(a_musicB(:,i)),a_musicB(:,i));
    end



    P_A = pinv(R_A_music)*R_A(:);
    P_B = pinv(R_B_music)*R_B(:);


    GAMMA = pinv(a_musicA*diag(sqrt(P_A)))*A;
    B_music_P = a_musicB*diag(sqrt(P_B));
    B_GMMMA = B*pinv(GAMMA);

    R_B_music_P = zeros(n*n,rhat);

    for i = 1:rhat
        R_B_music_P(:,i) = kron(B_music_P(:,i),B_music_P(:,i));
    end

    R_B_GMMMA = B_GMMMA*B_GMMMA.';

    exp2phi = pinv(R_B_music_P)*R_B_GMMMA(:);

    B_music_P_phi = B_music_P*diag(sqrt(exp2phi));

    permtx0 = pinv(B_music_P_phi)*B_GMMMA;

    row_index = 1:rhat;
    col_index = 1:rhat;
    record_index = zeros(rhat,2);
    permtx = zeros(size(permtx0));
    for i = 1: rhat
        value = max(max(abs(permtx0(row_index,col_index))));
        [index_max,index_min,~] = find(abs(permtx0(row_index,col_index))==value);
        permtx(row_index(index_max),col_index(index_min)) = 2*((real(permtx0(row_index(index_max),col_index(index_min)))>0)-0.5);
        record_index(i,:) = [row_index(index_max),col_index(index_min)]; 
        row_index = setdiff(row_index,row_index(index_max));
        col_index = setdiff(col_index,col_index(index_min));
    end

%     sum(sum(abs(real(permtx0)-permtx)))/rhat
    if sum(sum(abs(real(permtx0)-permtx)))/rhat<2
        W1_sort = W1;
        % [W1_sort,ind_ascend] = sort(W1_sort,'ascend');
        W2_sort = abs(permtx')*W2;

        % W2_sort = W2_correct(ind_ascend);
        % W2_sort = W2_correct;
    %     theta_hat = [W1_sort,W2_sort];


        A_m_hat = Array_fun(m,W1_sort);
        B_n_hat = Array_fun(n,W2_sort);
        HH_hat = zeros(m*n,rhat);
        for i = 1:rhat
            HH_hat(:,i) = kron(conj(B_n_hat(:,i)),A_m_hat(:,i));
        end
        g_hat = HH_hat\Hhat(:);

        H_hat_MUSIC = A_m_hat*diag(g_hat)*B_n_hat';

%         norm(H_hat_MUSIC-Hhat,'fro')
        NMSE_MUSIC = 20*log10(norm(H_hat_MUSIC-H,'fro')/norm(H,'fro'));
    else          % Recovered frequencies are very bad
        NMSE_MUSIC = NMSE(end);
    end
end

