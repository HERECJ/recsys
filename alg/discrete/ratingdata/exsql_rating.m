function [B,D]=exsql_rating(R,varargin)
[opt.lambda,opt.beta,opt.e,max_iter,k, test, debug] = process_options(varargin,'lambda',0.1,'beta',1,'e',0,'max_iter',10,'K',20,'test',[],'debug',true);
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
    %B = optimize_P(Rt,D,B,Nu,opt);
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
            
            para_0 = opt.beta * n + (2 * opt.e - opt.beta) * Nu(i);
            L0 = para_0 * sum(r_square);
            L1 = -2 * para_0 * b * rq;
            b_2l = opt.beta - 2 * opt.lambda;
            L2 = ( b_2l * n + (2 * opt.e - 2 * opt.beta + 2* opt.lambda) * Nu(i)) * (b*Du'*Du*b');
            L3 = (2 * opt.lambda * n + b_2l * Nu(i)) * b * DtD * b'; 
            L4 = -2 * opt.e * r_sum * r_sum;
            L5 = r_sum * b * ((4 * opt.e - 2 * opt.beta) * qi+ 2*opt.beta * qsum);
            bqi = b * qi;
            bqq = b * qq; 
            L6 = -2 * opt.e * bqi * bqi - 2 * opt.beta * bqi * bqq - 2 * opt.lambda * bqq * bqq;

            v = v + L0 + L1 + L2 + L3 + L4 + L5 + L6;
        end
    end
end

function B = optimize_P(Rt,D,B,Nu,opt)
[n,m]=size(Rt);
DtD = D'*D;
%k=size(D,2);
qsum = sum(D,1)';
b_2l = opt.beta - 2 * opt.lambda;
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
    
    A0 = ( b_2l * n + (2 * opt.e - 2 * opt.beta + 2* opt.lambda) * Nu(u)) * (Du'*Du);
    A1 = ( 2 * opt.lambda * n + b_2l * Nu(u)) * DtD; 
    A2 = -2 * opt.e * (qu * qu') - opt.beta * qu * q' - opt.beta * q * qu'- 2 * opt.lambda * (q * q');
    
    para_0 = opt.beta * n + (2 * opt.e- opt.beta) * Nu(u);
    b0 = para_0 * rq;
    b1 = -r_sum * ((2 * opt.e-  opt.beta) * qu + opt.beta * qsum);

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
para0 = opt.beta - 2 * opt.lambda;
NN = 2 * opt.lambda * n + para0 * Nu;
NiPPt = B' * bsxfun(@times, NN, B);

para1 = 2 * opt.e + 2 * opt.lambda - 2 * opt.beta;
para2 = opt.beta * n + (2 * opt.e - opt.beta) * Nu;

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

PtP = B' * B;
PPtq = B' * r_;
R_rul = B' * RQ_;
for l=1:n
    rl = R(:,l);
    [~,~,rr] = find(rl);
    d = D(l,:)';
    idx = rl~=0;
    Bl = B(idx,:);
    Q_l = Q_(idx,:);
    Nu_l = Nu(idx);
    r_l = r_(idx);
    
    V= NiPPt + Bl' * bsxfun(@times, para0 * n + para1 * Nu_l, Bl) ;
    E1 = Bl' * (para2(idx) .* rr);
    E2 = para0 * PPtq + 2 * opt.lambda * PtP * dsum;
    E3 = para0 * Bl' * (Bl * dsum) + para1 * ( Bl' * r_l);
    E4 = - opt.beta * R_rul + (opt.beta - 2 * opt.e)* Bl' * RQ_(idx);
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
    PPtq = PPtq +  Bl' * (r_l_new - r_l);
end
end