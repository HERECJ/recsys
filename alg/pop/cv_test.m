function [eval_detail] = cv_test(mat, varargin)
% rec: recommendation method
% mat: matrix storing records
% mode of cross validation: 
%   un: user side normal 
%   in: item side normal
%   en: entry wise normal
%   u: user
%   i: item
[folds, fold_mode, seed,] = process_options(varargin, 'folds', 5, 'fold_mode', 'un', 'seed', 1);
assert(folds>0)
rng(seed);
mat_fold = kFolds(mat, folds, fold_mode);
eval_detail = cell(folds,1);
parfor i=1:folds
    testdata = mat_fold{i};
    traindata = mat - testdata;
    eval_detail{i} = pop_items(traindata,testdata, 'topk', 200, 'cutoff', 200);
end

