function [evalout] = pop_items(traindata,testdata,varargin)
[topk, cutoff] = process_options(varargin, 'topk', 200, 'cutoff', 200);
fprintf("pop baselines\n")
if topk > 0 && cutoff > 0
    topk = cutoff;
elseif cutoff<=0
    if topk>0
        cutoff = topk;
    else
        cutoff = 200;
    end
end
[M,N] = size(traindata);
Et = traindata.';
cand_count = N - sum(Et ~= 0);
cand_count = cand_count.';
items_cnt = sum(traindata~=0);

[~,index] = maxk(items_cnt,topk);
[v,i,j] = find(index);
user_list = cell(M, 1);
item_list = cell(M, 1);
val_list = cell(M, 1);
for u = 1:M
    user_list{u} = u * v;
    item_list{u} = j;
    val_list{u} = i;
end
mat_rank = sparse(cell2mat(user_list),cell2mat(item_list),cell2mat(val_list),M,N);

if isexplict(testdata)
    evalout = compute_rating_metric(testdata, mat_rank, cand_count, cutoff);
    fprintf('1\n');
else
    evalout = compute_item_metric(testdata, mat_rank, cand_count, cutoff);
    fprintf('2\n');
end
end