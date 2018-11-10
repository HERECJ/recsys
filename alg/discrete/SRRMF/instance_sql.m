delete(gcp('nocreate'));
clear;
% read data
addpath(genpath('~/rec/recsys'))  % change the path
dir= '~/rec/newdata';
num_threads = 4; 
%dataset = 'yelpdata';
%dataset = 'amazondata';
dataset = 'netflixdata';
P = parpool('local',num_threads); 
P.IdleTimeout = 60000;

load(sprintf('%s/%s.mat',dir,dataset));
if ~exist('data','var')
%    Traindata(Testdata>0) = 0;
    data = data;
end

K = [8,16,32,64,128];

if ~exist('result','var')
    sql = cell(length(K),1);
end

for i = 1:length(K)
    fprintf('the dimension of latent vector K: %d\n', K(i))
    
    algs.alg = @(mat,varargin) sql_fresh(mat, 'max_iter', 10, 'K', K(i), 'debug',true, varargin{:});
    algs.paras = {'rating',true,'lambda',[0,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10]};
    
    [outputs{1:6}] = running(algs.alg, data, algs.paras{:});
    sql{i} = outputs;
    save('~/rec/results/netflix/sql_netflix.mat', 'sql');
end 
