function [B,D]=exsql(R,varargin)
[opt.lambda,max_iter,k, test, debug] = process_options(varargin,'lambda',0.1,'max_iter',10,'K',20,'test',[],'debug',true);
print_info()
[m,n]=size(R);
Rt=R';
rng(10);
B=randn(m,k);
D = randn(n,k);
Nu = sum(R~=0,2);
converge = false;
loss0 = 0;
it = 1; % the number of iteration

%the converge process
while ~converge
    D = optimize_Q(R,B,D,Nu,opt);
    B = optimize_P(Rt,D,B,Nu,opt);
    %D = optimize_Q(R,B,D,Nu,opt);
    loss = loss_();
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.1f,', it, loss);
        if ~isempty(test)
            metric = evaluate_rating(test, B, D, 10);
            fprintf('ndcg@1=%.3f', metric.rating_ndcg(1));
        end
        fprintf('\n')
    end
    if it >= max_iter ||abs(loss0-loss)<1 || (it >1 && (loss-loss0)>1e-6)
        converge = true;
    end
    it = it + 1;
    loss0 = loss;
end

    function print_info()
        fprintf('implicit feedback (K=%d,max_iter=%d,lambda=%f)\n',k,max_iter,opt.lambda);
    end

    function v = loss_()
        v = 0;
        qsum = sum(D,1)';
        DtD = D'*D;
        for i = 1:m
            %loss
            r = Rt(:,i);
            b = B(i,:);
            idx = r ~= 0;
            Du = D(idx,:);
            
            qi = sum(Du,1)'; %q_i~
            qq = qsum - qi; %q~-q_i~
            
            L1 = n*(b*Du'*Du*b') +  Nu(i)*(b*DtD*b')+n*Nu(i);
            L2 = -2*n*b*qi + 2*Nu(i)*b*qsum - 2*(b*qi)*(b*qsum);
            
            %regular
            R1 = (n-Nu(i))*b*( DtD - Du'*Du )*b';
            R2 = -(b*qq)*(b*qq);
            
            v = v + (L1 + L2) + opt.lambda*(R1 + R2);
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
    idx = r ~= 0;
    Du = D(idx,:);
    qu = sum(Du,1)'; %q_i~        ;
    q = qsum - qu; %q~-q_i~
    
    Qu= DtD - Du'*Du;
    A1 = Nu(u)*(DtD) + n*(Du'*Du) - qu*qsum'-qsum*qu';
    A2 = opt.lambda*((n-Nu(u))*Qu - q*q');
    A = A1 + A2;
    b = n*qu - Nu(u)*qsum;
    %dA = decomposition(A);
    B(u,:) = A\b;
end
end

function D = optimize_Q(R,B,D,Nu,opt)
[m,n] = size(R);
Rt = R';
k = size(D,2);
%N = zeros(m,1) + n;
%N = n * ones(m,1);
%opt.dsum = sum(D,1)';
dsum = sum(D,1)';
Np = B'*Nu;

% compute the Q~ matrix
% Q~ : M*K   M = the number of users
% the colunm of Q~ is the qi~ (the sum of qj where j is in E_i)
NiPPt = B' * bsxfun(@times, Nu, B);
%NiPPT1 = B' * (repmat(Nu, 1, k) .* B);
N_PPt = B' * bsxfun(@times, n - Nu, B);
Q_ = zeros(m,k);
for u = 1:m
    r = Rt(:,u);
    idx = r~=0;
    Du = D(idx,:);
    Q_(u,:) = sum(Du,1);
end
r_ = sum(B .* Q_, 2);

%PPtq = B'*sum(DotBQ,2);
%pptqq = B'*sum(DotBQQ,2);
PtP = B' * B;
PPtq = B' * r_;
for l=1:n
    rl = R(:,l);
    d = D(l,:)';
    idx = rl~=0;
    Bl = B(idx,:);
    Q_l = Q_(idx,:);
    Nu_l = Nu(idx);
    r_l = r_(idx);
    
    V = opt.lambda * N_PPt + NiPPt + Bl'*bsxfun(@times, ...
        (1-opt.lambda)*n - (-opt.lambda)*Nu_l, Bl);
    %V1 = NiPPt + Bl'*bsxfun(@times, n, Bl);
    %V2 = opt.lambda * (N_PPt - Bl'*bsxfun(@times, n-Nu_l, Bl));
    
    E1 = n*sum(Bl,1)'-Np;
    %E2 = PPtq + Bl'*sum(Bl.*(dsum-2*Q_l)',2);
    E2 = (1 - opt.lambda)*PPtq + ((1 - opt.lambda) * (Bl' * (Bl * dsum))) -(- opt.lambda)*( Bl' * r_l);
    E3 = opt.lambda * (PtP * dsum);
    %V = V1 +V2;
    E = E1+E2+E3;
    
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
    PPtq = PPtq + Bl' * (r_l_new - r_l);
    
end
end