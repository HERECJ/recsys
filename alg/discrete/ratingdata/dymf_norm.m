function [B,D]=dymf_norm(R,varargin)
%Dynamic Matrix Factorization with Priors on Unknown Values
    [opt.lambda,opt.sigma,max_iter,k, test, debug] = process_options(varargin,'lambda',0.1,'sigma',0.0,'max_iter',20,'K',20,'test',[],'debug',true);
    print_info()

    [m,n]=size(R);
    Rt=R';
    rng(200);
    B = randn(m,k);
    D = randn(n,k);
    Nu = sum(R~=0,2);
    converge = false;
    loss0 = 0;
    it = 1; % the number of iteration
    
    
    %the converge process
    while ~converge
        D = optimize_Q(R,B,D,Nu,opt);
        B = optimize_P(Rt,D,B,Nu,opt);
        loss = loss_();
        if debug 
            fprintf('Iteration=%3d of all optimization, loss=%.4f,', it, loss);
            if ~isempty(test) 
                metric = evaluate_rating(test, B, D, 10);
                fprintf('ndcg@1=%.3f', metric.rating_ndcg(1));
            end
            fprintf('\n')
        end
%         if it >= max_iter || abs(loss0-loss)<1e-6 * loss 
        if it >= max_iter ||abs(loss0-loss)<1 || (it >1 && (loss-loss0)>1e-6)
%         if it>= max_iter
            converge = true;
        end
        it = it + 1;
        loss0 = loss; 
    end
   
    function print_info()
        fprintf('explicit feedback (K=%d,max_iter=%d,lambda=%f,sigma=%f)\n',k,max_iter,opt.lambda,opt.sigma);
    end 
    
    %the computation of loss
    function v = loss_()
        v = 0;
        %qsum = sum(D,1)';
        DtD = D'*D;
        for i = 1:m
            %loss
            r = Rt(:,i);
            [~,~,rr] = find(r);
            b = B(i,:);
            idx = r ~= 0;
            Du = D(idx,:);
            
            %loss
            %qi = sum(Du,1)'; %q_i~ 
            qi = Du' * rr;  
            ll = b * Du'* Du * b';
            L1 = ll - 2 * b * qi + sum(rr.^2);
            
            
            %regular
            L2 = b * DtD * b' - ll;

            v = v + L1 / Nu(i) + opt.lambda * L2 / (n - Nu(i));            
        end
        v = v + opt.sigma * (norm(B) + norm(D));
    end 
end

function B = optimize_P(Rt,D,B,Nu,opt)
    [n,m]=size(Rt);
    DtD = D'*D;
    for u = 1:m
        r = Rt(:,u);
        [~,~,rr] = find(r);
        %B(u,:)
        idx = r ~= 0;
        Du = D(idx,:);
        %qu = sum(Du,1)'; %q_i~        ;
        
        A1 = Du' * Du;
        b = (1 / Nu(u)) *  Du' * rr  + opt.sigma;
        
        A2 = DtD - A1;
        A = A1 / Nu(u) + (opt.lambda / (n - Nu(u))) *  A2;
        % 0.01  is the 2-norm of the vector
        B(u,:) = A\b;
    end     
end

function D = optimize_Q(R,B,D,Nu,opt)
    [~,n] = size(R); 
    NNu = 1 ./ Nu;
    NNuu = 1 ./(n - Nu);
    BtB = B' * bsxfun(@times,NNuu,B);
    for l = 1 : n
        r = R(:,l);
        [~,~,rr]  = find(r);
        idx = r > 0;
        Bl = B(idx,:);
        
        A1 = Bl' * bsxfun(@times,NNu(idx),Bl);
        b = Bl' *(rr .* NNu(idx)) +  opt.sigma;
        A2 = BtB - Bl' * bsxfun(@times,NNuu(idx),Bl); 
        A = A1 + opt.lambda * A2 ;
        D(l,:) = A\b;
    end
   
end