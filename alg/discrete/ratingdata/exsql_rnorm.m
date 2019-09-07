function [B,D]=exsql_rnorm(R,varargin)
[opt.lambda,opt.beta,opt.e,max_iter,k, test, debug] = process_options(varargin,'lambda',0.1,'beta',1,'e',1,'max_iter',10,'K',20,'test',[],'debug',true);
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
    loss = loss_();
    D = optimize_Q(R,B,D,Nu,opt);
    B = optimize_P(Rt,D,B,Nu,opt);
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
            nu_sq = Nu(i)^0.5;
            nnu_sq = (n - Nu(i))^0.5;
            nu_nnu = nu_sq / nnu_sq;
            nnu_nu = nnu_sq / nu_sq;
            qi = sum(Du,1)'; %q_i~
            qq = qsum - qi; %q~-q_i~
            
            r_square = rr .^2;
            rq = Du' * rr;
            r_sum = sum(rr);
            ee = opt.e / Nu(i);
            beta = opt.beta /(nu_sq * nnu_sq);
            lambda = opt.lambda / (n - Nu(i));
            para_0 = 2 * opt.e + opt.lambda * nnu_nu;
            L0 = para_0 * sum(r_square);
            L1 = -2 * para_0 * b * rq;
            b2e =  opt.beta * nu_nnu + 2 * opt.lambda;
            L2 = ( para_0 - b2e )* (b*Du'*Du*b');
            L3 = b2e * b * DtD * b'; 
            L4 = -2 * ee * r_sum * r_sum;
            L5 = r_sum * b * ((4 * ee  - 2 * beta) * qi + 2 * beta * qsum);
            bqi = b * qi;
            bqq = b * qq; 
            L6 = -2 * ee * bqi * bqi - 2 * beta * bqi * bqq - 2 * lambda * bqq * bqq;

            v = v + L0 + L1 + L2 + L3 + L4 + L5 + L6;
        end
    end
end

function B = optimize_P(Rt,D,B,Nu,opt)
[n,m]=size(Rt);
DtD = D'*D;
%k=size(D,2);
qsum = sum(D,1)';

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
    
    nu_sq = Nu(u)^0.5;
    nnu_sq = (n - Nu(u))^0.5;
    nu_nnu = nu_sq / nnu_sq;
    nnu_nu = nnu_sq / nu_sq;
    ee = opt.e / Nu(u);
    beta = opt.beta / (nu_sq * nnu_sq);
    lambda = opt.lambda / (n - Nu(u));
    para_0 = 2 * opt.e + opt.beta * nnu_nu;
    b2e =  opt.beta * nu_nnu + 2 * opt.lambda;
    A0 = ( para_0 - b2e ) * (Du'*Du);
    A1 = b2e * DtD; 
    A2 = -2 * ee * (qu * qu') - beta * qu * q' - beta * q * qu'- 2 * lambda * (q * q');
  
    b0 = para_0 * rq;
    b1 = -r_sum * ((2 * ee - beta) * qu + beta * qsum);

    A = A0 + A1 + A2 ;
    b = b0 + b1 ;
    %dA = decomposition(A);
    B(u,:) = A\b;
end
end

function D = optimize_Q(R,B,D,Nu,opt)
[m,n] = size(R);
Rt = R';
k = size(D,2);
dsum = sum(D,1)';

nu_sq = Nu.^0.5;
nnu_sq = (n - Nu).^0.5;
nu_nnu = nu_sq ./ nnu_sq;
nnu_nu = nnu_sq ./ nu_sq;
ee = opt.e ./ Nu;
beta = opt.beta ./ (nu_sq .* nnu_sq);
lambda = opt.lambda ./ ( n - Nu );


NN = 2 * opt.lambda + opt.beta * nu_nnu;
NiPPt = B' * bsxfun(@times, NN, B);

para1 = 2 * ee + 2 * lambda - 2 * beta;
para2 = 2 * opt.e + opt.beta * nnu_nu;

% compute the Q~ matrix
% Q~ : M*K   M = the number of users
% the colunm of Q~ is the qi~ (the sum of qj where j is in E_i)
Q_ = zeros(m,k);
RQ_ = zeros(m,1);
for u = 1:m
    r = Rt(:,u);
    [~,~,rru] = find(r);
    idx = r~=0;
    Du = D(idx,:);
    Q_(u,:) = sum(Du,1);
    RQ_(u) = sum(rru);
end
r_ = sum(B .* Q_, 2);
PtP = bsxfun(@times,2 * lambda,B)' * B;

para0 = beta - 2 * lambda ;

PPtq = bsxfun(@times,para0,B)' * r_;
R_rul = bsxfun(@times,beta,B)' * RQ_;
for l=1:n
    rl = R(:,l);
    [~,~,rr] = find(rl);
    d = D(l,:)';
    idx = rl~=0;
    Bl = B(idx,:);
    Q_l = Q_(idx,:);
    r_l = r_(idx);
    
    V= NiPPt + Bl' * bsxfun(@times, ( para2(idx) - NN(idx)), Bl) ;
    E1 = Bl' * (para2(idx) .* rr);
    E2 = PPtq + PtP * dsum;
    
    E3 = bsxfun(@times,para0(idx),Bl)' * (Bl * dsum) + bsxfun(@times,para1(idx),Bl)' * r_l;
    E4 = - R_rul + bsxfun(@times,beta(idx)-2 * ee(idx),Bl)' * RQ_(idx);
    E = E1 + E2 + E3 + E4;
    dl = V\E;
    D(l,:) = dl;
    
    % update dsum
    dsum = dsum - d + dl;
        % update Q_
    Q_l_new = Q_l + (dl - d)';
    Q_(idx,:) = Q_l_new;
    % update r_
    r_l_new = r_l + sum(Bl .* (Q_l_new - Q_l), 2);
    r_(idx) = r_l_new;
    % update PPtq
    PPtq = PPtq +   bsxfun(@times,para0(idx),Bl)' * (r_l_new - r_l);
end
end