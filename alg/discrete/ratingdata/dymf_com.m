function [B,D]=dymf_com(R,varargin)
%loss: the rating regression
%regularizer: comparison
    [opt.lambda,opt.sigma,max_iter,k, test, debug] = process_options(varargin,'lambda',0.1,'sigma',0.01,'max_iter',20,'K',20,'test',[],'debug',true);
    print_info()

    [m,n]=size(R);
    Rt=R';
    rng(10);
    B = randn(m,k);
    D = randn(n,k);
    Nu = sum(R~=0,2);
    converge = false;
    loss0 = 0;
    it = 1; % the number of iteration
    
    %the converge process
    while ~converge
        B = optimize_P(Rt,D,B,Nu,opt);
        D = optimize_Q(R,B,D,Nu,opt);
        loss = loss_();
        if debug 
            fprintf('Iteration=%3d of all optimization, loss=%.4f,', it, loss);
            if ~isempty(test) 
                metric = evaluate_rating(test, B, D, 10);
                fprintf('ndcg@1=%.3f', metric.rating_ndcg(1));
            end
            fprintf('\n')
        end
        if it >= max_iter || abs(loss0-loss)<1e-6 * loss 
        %if it>= max_iter 
            converge = true;
        end
        it = it + 1;
        loss0 = loss; 
    end
   
    function print_info()
        fprintf('explicit feedback (K=%d,max_iter=%d,lambda=%f)\n',k,max_iter,opt.lambda);
    end 
    
    %the computation of loss
    function v = loss_()
        v = 0;
        qsum = sum(D,1)';
        DtD = D'*D;
        for i = 1:m
            %loss
            r = Rt(:,i);
            [~,~,rr] = find(r);
            b = B(i,:);
            idx = r ~= 0;
            Du = D(idx,:);
            
            %loss
            qi = sum(Du,1)'; %q_i~ 
            ll = b * Du'* Du * b';
            L1 = ll - 2 * b * (Du' * rr) + sum(rr.^2);
            
            %regular
            qq = qsum - qi; %q~-q_i~ 
            R1 = ( n - Nu(i)) * b * ( DtD - Du' * Du ) * b';
            R2 = -( b * qq) * (b*qq);
            
            v = v + L1 + opt.lambda * (R1 + R2);            
        end
        v = v + opt.sigma * (norm(B) + norm(D));
    end 
end

function B = optimize_P(Rt,D,B,Nu,opt)
    [n,m]=size(Rt);
    qsum = sum(D,1)';
    DtD = D'*D;
    for u = 1:m
        r = Rt(:,u);
        [~,~,rr] = find(r);
        idx = r ~= 0;
        Du = D(idx,:);
        qu = sum(Du,1)'; %q_i~        ;

        A1 = Du' * Du;
        b1 = Du' * rr;
       
        qq = qsum - qu;  % k*1 vector 
        v = (n - Nu(u)) * (DtD - Du' * Du) -  qq * qq';
        A2 = opt.lambda  * v;
        A =  A1 + A2 + opt.sigma;
        b =  b1;
        B(u,:) = A\b;
    end     
end

function D = optimize_Q(R,B,D,Nu,opt)
    [m,n] = size(R);
    Rt = R';
    dsum = sum(D,1)';
    
    k = size(D,2);

    Pt = B' * bsxfun(@times , n - Nu , B);
    Q_ = zeros(m,k);
    for u = 1:m
        r = Rt(:,u);
        idx = r>=0;
        Du = D(idx,:);
        Q_(u,:) = sum(Du,1);
    end
        
    r_ = sum(B .* Q_, 2); 
    %the updata of q_l
    
    PtP = B' * B;
    Pq = PtP * dsum - B' * r_;

    for l = 1 : n
        r = R(:,l);
        [~,~,rr]  = find(r);
        d = D(l,:);
        idx = r > 0;
        Bl = B(idx,:);
        Q_l = Q_(idx,:);
        r_l = r_(idx);

        A1 = Bl' * Bl;
        b1 = Bl' * rr ;
        
        Ptl = Bl'*bsxfun(@times, n - Nu(idx) , Bl);
        Pql = Bl' * (Bl * dsum) - Bl' * r_l;
        A2 = opt.lambda * (Pt - Ptl);
        b2 = opt.lambda * (Pq - Pql);
        
        A = A1 + A2 +opt.sigma;
        b = b1 + b2;
        dl = A\b;
        D(l,:) = dl;
        
        % update dsum
        dsum = dsum - d' + dl; 
        % update Q_
        Q_l_new = Q_l + (dl' - d);
        Q_(idx,:) = Q_l_new;
        % update r_
        r_l_new = r_l + sum(Bl .* (Q_l_new - Q_l), 2);
        r_(idx) = r_l_new;
        % update PPtq
        Pq = Pq + Bl' * (r_l_new - r_l);      
    end  
end