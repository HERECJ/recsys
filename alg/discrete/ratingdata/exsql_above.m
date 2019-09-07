function [B,D] = exsql_above(R,varargin)
[opt.lambda,opt.beta,opt.e,max_iter,k, test, debug] = process_options(varargin,'lambda',0,'beta',0.1,'e',1,'max_iter',10,'K',20,'test',[],'debug',true);
print_info();
[m,n]=size(R);
Rt=R';
RR = avr_gen(R);
RRt = RR';
rng(10);
B = randn(m,k);
D = randn(n,k);
Nu = sum(R~=0,2);
abv_Nu = sum(RR~=0,2);
converge = false;
loss0 = 0;
it = 1; % the number of iteration

%the converge process
while ~converge
    loss = loss_();
    D = optimize_Q(R,RR,B,D,Nu,abv_Nu,opt);
    B = optimize_P(Rt,RRt,D,B,Nu,abv_Nu,opt);
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.8f,', it, loss);
        if ~isempty(test)
            metric = evaluate_rating(test, B, D, 10);
            fprintf('ndcg@1=%.3f', metric.rating_ndcg(1));
        end
        fprintf('\n')
    end
     if it >= max_iter 
%    if it >= max_iter ||abs(loss0-loss)<1e-4 || (it >1 && (loss-loss0)>1e-6)
        converge = true;
    end
    it = it + 1;
    loss0 = loss;
end

    function print_info()
        fprintf('explicit feedback (K=%d,max_iter=%d,beta=%f,lambda=%f)\n',k,max_iter,opt.beta,opt.lambda);
    end

    function v = loss_()
        v = 0;
        qsum = sum(D,1)';
        DtD = D'*D;
        for i = 1:m
            %loss
            r = Rt(:,i);
            b = B(i,:);
            [~,~,rr]=find(r);
            idx = r ~= 0;
            Du = D(idx,:);
            
            qi = sum(Du,1)'; %q_i~
            qq = qsum - qi; %q~-q_i~
            
            r_square = rr .^2;
            rq = Du' * rr;
            r_sum = sum(rr);
            
            para_0 =  (2 * opt.e ) * Nu(i);
            L0 = para_0 * sum(r_square);
            L1 = -2 * para_0 * b * rq;
            b_2l = - 2 * opt.lambda;
            L2 = ( b_2l * n + (2 * opt.e  + 2* opt.lambda) * Nu(i) - opt.beta * abv_Nu(i) ) * (b*Du'*Du*b');
            L3 = (2 * opt.lambda * n + b_2l * Nu(i) + opt.beta * abv_Nu(i)) * b * DtD * b'; 
            L4 = -2 * opt.e * r_sum * r_sum;
            L5 = r_sum * b * ((4 * opt.e) * qi);
            bqi = b * qi;
            bqq = b * qq; 
            L6 = -2 * opt.e * bqi * bqi  - 2 * opt.lambda * bqq * bqq;
            L = L0 + L1 + L2 + L3 + L4 + L5 + L6;
            
            r_ = RRt(:,i);
            [~,~,rr_] = find(r_);
            idx_ = rr_ ~= 0;
            DDu = D(idx_,:);
            quu = sum(DDu,1)';
            rr_sq = rr_ .^2;
            rr_q = DDu' * rr_;
            rr_sum = sum(rr_);
            para_r0 = n - Nu(i);
            LL0 = para_r0 * sum(rr_sq);
            LL1 = - 2 * para_r0 * b * rr_q;
            LL2 = para_r0 * (b * DDu' * DDu * b');
            LL3 = 2 * rr_sum * b * qq - 2 * (b * quu) * (b * qq);
            LL = opt.beta * (LL0 + LL1 + LL2 + LL3);
            
            v = v + L + LL;
        end
    end
end

function B = optimize_P(Rt,RRt,D,B,Nu,abv_Nu,opt)
[n,m]=size(Rt);
DtD = D'*D;
%k=size(D,2);
qsum = sum(D,1)';
b_2l =  - 2 * opt.lambda;
for u = 1:m
    r = Rt(:,u);
    %b = B(u,:);
    [~,~,rr]=find(r);
    idx = r ~= 0;
    Du = D(idx,:);
    qu = sum(Du,1)'; %q_i~        ;
    q = qsum - qu; %q~-q_i~
    
    rq = Du' * rr;
    r_sum = sum(rr);
    
    A0 = ( b_2l * n + (2 * opt.e  + 2* opt.lambda) * Nu(u) - opt.beta * abv_Nu(u)) * (Du'*Du);
    A1 = ( 2 * opt.lambda * n + b_2l * Nu(u) + opt.beta * abv_Nu(u)) * DtD; 
    A2 = -2 * opt.e * (qu * qu') - 2 * opt.lambda * (q * q');
    
    para_0 = (2 * opt.e) * Nu(u);
    b0 = para_0 * rq;
    b1 = -r_sum * ((2 * opt.e) * qu );
    
    r_ = RRt(:,u);
    [~,~,rr_] = find(r_);
    idx_ = rr_ ~= 0;
    DDu = D(idx_);
    quu = sum(DDu,1)';
    rr_q = DDu' * rr_;
    rr_sum = sum(rr_);
    para_r0 = n - Nu(u);
    
    A3 = para_r0 * (DDu' * DDu) -  quu * q' - q * quu';
    A = A0 + A1 + A2 + opt.beta * A3;
    
    b2 = para_r0 * rr_q - rr_sum * q;
    b = b0 + b1 + opt.beta * b2;
    
    %dA = decomposition(A);
    B(u,:) = A\b;
end
end

function D = optimize_Q(R,RR,B,D,Nu,abv_Nu,opt)
[m,n] = size(R);
Rt = R';
RRt = RR';
k = size(D,2);
dsum = sum(D,1)';
para0 =  - 2 * opt.lambda;
NN = 2 * opt.lambda * n + para0 * Nu + opt.beta * abv_Nu;
NiPPt = B' * bsxfun(@times, NN, B);

para1 = 2 * opt.e + 2 * opt.lambda ;
para2 =  (2 * opt.e) * Nu;

% compute the Q~ matrix
% Q~ : M*K   M = the number of users
% the colunm of Q~ is the qi~ (the sum of qj where j is in E_i)
Q_ = zeros(m,k);
RQ_ = zeros(m,1);

QQ_ = zeros(m,k);
RQQ_ = zeros(m,1);
for u = 1:m
    r = Rt(:,u);
    [~,~,ru] = find(r);
    idx = r~=0;
    Du = D(idx,:);
    Q_(u,:) = sum(Du,1);
    RQ_(u) = sum(ru);
    
    rr = RRt(:,u);
    [~,~,rru ] = find(rr);
    idx_r = rru ~= 0;
    DDu = D(idx_r,:);
    QQ_(u,:) = sum(DDu,1);
    RQQ_(u,:) = sum(rru);
end
r_ = sum(B .* Q_, 2);
rr_ = sum(B .* QQ_,2);

PtP = B' * B;
PPtq = B' * r_;
PLtq = B' * rr_;

R_rul = B' * RQ_;
RR_rul = B' * RQQ_;
for l=1:n
    rl = R(:,l);
    [~,~,rr] = find(rl);
    d = D(l,:)';
    idx = rl~=0;
    Bl = B(idx,:);
    Q_l = Q_(idx,:);
    r_l = r_(idx);
    
    rl_ = RR(:,l);
    [~,~,rr_l] = find(rl_);
    idx_rl = rr_l ~= 0;
    Bl_ = B(idx_rl,:);
    QQ_l = QQ_(idx_rl,:);
    rr_ll = rr_(idx_rl);
    
    V0 = NiPPt + Bl' * bsxfun(@times, para0 * n + para1 * Nu(idx) - opt.beta * abv_Nu(idx), Bl) ;
    V1 = opt.beta * Bl_' * bsxfun(@times, n - Nu(idx_rl),Bl_);
    V = V0 + V1;
    
    E1 = Bl' * (para2(idx) .* rr);
    E2 = para0 * PPtq + 2 * opt.lambda * PtP * dsum;
    E3 = para0 * Bl' * (Bl * dsum) + para1 * ( Bl' * r_l);
    E4 = - opt.lambda * R_rul + (- 2 * opt.e)* Bl' * RQ_(idx);
    E0 = E1 + E2 + E3 + E4;
    
    EE1 = Bl_' * ((n - Nu(idx_rl)) .* rr_l); 
    EE2 = PLtq + Bl_' * (Bl_ * dsum) -  Bl_' * r_(idx_rl) - Bl' * rr_(idx);
    EE3 = - RR_rul + Bl_' * RQQ_(idx_rl);
    EE0 = EE1 + EE2 + EE3;
    
    E = E0 + opt.beta * EE0;
    dl = V\E;
    D(l,:) = dl;
    
    % update dsum
    dsum = dsum - d + dl;
    % update Q_
    Q_l_new = Q_l + (dl - d)';
    Q_(idx,:) = Q_l_new;
    
    QQ_l_new  = QQ_l +  (dl - d)';
    QQ_(idx_rl,:) = QQ_l_new;
    % update r_
    r_l_new = r_l + sum(Bl .* (Q_l_new - Q_l), 2);
    r_(idx) = r_l_new;
    
    rr_ll_new = rr_ll + sum(Bl_ .* (QQ_l_new - QQ_l), 2);
    rr_(idx_rl) = rr_ll_new;
    
    % update PPtq
    PPtq = PPtq +  Bl' * (r_l_new - r_l);
    PLtq = PLtq + Bl_' * (rr_ll_new - rr_ll);
end
end