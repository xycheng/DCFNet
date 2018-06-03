% This code in matlab is for demonstration use only, and the efficiency is 
% NOT optimized. The theoretical computation reduction of DCF Net is not 
% achieved, and the code is slower than a regular CNN with the same depth 
% and width. The memory cost is K times of the regular CNN. 
% 
% To run this code, make sure that 
%
%   - the matconvnet library (http://www.vlfeat.org/matconvnet/) is 
%     installed, and the binaries in "matlab" is under the folder named 
%     "matconvnet" in the root directory of the codes. Make sure that the 
%     mex files in "matconvnet/matlab/mex/" are compatible with the 
%     platform and the matlab version in use.
%
%   - the dataset .mat file is in the "data/" folder. For this demo code
%     with mnist, the data/mnist.mat file should include three variables
%     "data" 28x28x1x70000, "labels" 1x70000. "data" include mnist samples
%     where each pixel takes value from 0 to 255
%

clear;clc;
run('./matconvnet/matlab/vl_setupnn.m');

%% load data

disp('... load data ...')
load(['./data/mnist.mat']);

imdb.images.labels=labels;
imdb.images.data = single(data)/255;
clear data set labels;

disp('done.')

%% train param

paras.GPU = 0; % to use GPU, install matconvnet binaries with GPU, and set to 1

paras.B = 100; % batch size
paras.m = 0.9; %momentum
paras.w = 1e-4; %weight decay 

paras.snapshot = 10; 
paras.teststep = 1; 

%
paras.eta  = logspace(-2, -4, 100); % learning rate, set number of steps to 
                                    % be 100, to obtain a testing accuracy 
                                    % ~ 99.35 with ntr = 50K training data
                                    
paras.E = numel(paras.eta); %number of epoches



%% network parameter 

% bases
K = 3;

L1 = 2; %5x5
[psi, c, kq_Psi]= calculate_FB_bases(L1);

%
p=psi(:,1:K);
[u,s,v] = svd(p,'econ');
pinv = u * diag(1./diag(s)) * v';


% hyper parameter
M1_0  = 16;  %number of channels in 1st conv layer 
M2_fc = 128; %width of fc layer

%% network initialization

rng(2018);


%
f = 0.01;

net.layers = {} ;

% Block 1
M1=1;
M2=M1_0;

L = 5;

w1 = reshape(single(p), [L,L,1,K]);
b1 = zeros(1, K, 'single');
net.layers{end+1} = struct('type', 'conv-psi', ...
    'weights', {{ w1, b1}}, ...
    'stride', 1,...
    'pad', 0,...
    'learningRate', [0,0]);

w2 = f*randn(1,1,K*M1,M2,'single');
b2 = zeros(1, M2, 'single');
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{ w2, b2}}, ...
    'learningRate', [1,2]);

% batch normalization
net.layers{end+1} = struct('type', 'bnorm',...
            'weights', {{ones(M2, 1, 'single'), zeros(M2, 1, 'single'), ...
                        [zeros(M2, 1, 'single'),ones(M2, 1, 'single')]}}, ...
            'learningRate', [1 1 0.05], ...
            'weightDecay', [0 0]) ;

net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ; %24->12


% Block 2
M1=M2;
M2=M1_0*2;

w1 = reshape(single(p), [L,L,1,K]);
b1 = zeros(1, K, 'single');
net.layers{end+1} = struct('type', 'conv-psi', ...
    'weights', {{ w1, b1}}, ...
    'stride', 1,...
    'pad', 0,...
    'learningRate', [0,0]);

w2 = f*randn(1,1,K*M1,M2,'single');
b2 = zeros(1, M2, 'single');
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{ w2, b2}}, ...
    'learningRate', [1,2]);

% batch normalization
net.layers{end+1} = struct('type', 'bnorm',...
            'weights', {{ones(M2, 1, 'single'), zeros(M2, 1, 'single'), ...
                        [zeros(M2, 1, 'single'),ones(M2, 1, 'single')]}}, ...
            'learningRate', [1 1 0.05], ...
            'weightDecay', [0 0]) ;

net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ; %8->4


% Block 4, fc
M1=M2;
M2=M2_fc;

L = 4;

net.layers{end+1} = struct('type','conv',...
    'weights',{{ f*randn(L,L,M1,M2,'single'), zeros(1, M2, 'single') }}, ...
    'weightDecay',  [1,1],...
    'learningRate', [1,2]);

net.layers{end+1} = struct('type','relu'); %4->1

net.layers{end+1} = struct('type','dropout','rate',0.5);

% Block 5, fc
M1=M2;
M2=10;

net.layers{end+1} = struct('type','conv',...
    'weights',{{ f*randn(1,1,M1,M2,'single'), zeros(1, M2, 'single') }}, ...
    'weightDecay', [1,1],...
    'learningRate', [1,2]);

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.meta.inputSize = [28 28 1] ;

% Fill in default values
net = vl_simplenn_dcf_tidy(net) ;

%% split training testing

ntr  = 50000; 
nval = 1000;
nte  = 10000;
        
%% split training and testing
nmax = size(imdb.images.data,4);

idx = randperm(nmax,ntr+nval+nte);

imdb1.images.data = imdb.images.data(:,:,:,idx);
imdb1.images.labels = imdb.images.labels(idx);
imdb1.images.set = [ones(1,ntr),2*ones(1,nval),3*ones(1,nte)];

%
trainX = imdb.images.data(:,:,:,idx( 1:ntr ));
trainLabel = imdb.images.labels(idx( 1:ntr ));

valX = imdb.images.data(:,:,:,idx( ntr+1: ntr+nval ));
valLabel =  imdb.images.labels(idx( ntr+1: ntr+nval ));

testX = imdb.images.data(:,:,:, idx( ntr+nval+1: ntr+nval+nte ) );
testLabel = imdb.images.labels( idx( ntr+nval+1: ntr+nval+nte ) );

%
paras.expDir = sprintf('mnist_dcf_K%d_M%d_fc%d_NTr%d',K,M1_0, M2_fc,ntr);
paras.testlabel = valLabel;
paras.testdat = valX;

%% training

[net,info] = softmaxTrain7(trainX,trainLabel,net,paras);
plotStats(info,paras);

%% testing
paras_te = paras;

paras_te.testlabel = testLabel;
paras_te.testdat = testX;

disp('... testing error  ...')
tic
[~,info2]= softmaxTest7(net,paras_te);
toc

trainLoss = info.train(end).objective;
testLoss = info2.testLoss;
genGap =  double(testLoss - trainLoss)
        