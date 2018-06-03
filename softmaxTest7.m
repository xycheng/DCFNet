function [net, info] = softmaxTest7(net,paras)
% implement a deep convolutional or fully connected network that has same
% spirit as deepface
% The top layer is a softmaxloss layer by default

rng(0);

if ~isfield(paras,'eta')
    paras.eta = 0.01;
end

if ~isfield(paras,'B')
    paras.B = 100;
end

if ~isfield(paras,'E')
    paras.E = 50;
end

if ~isfield(paras,'w')
    paras.w = 0;
end

if ~isfield(paras,'m')
    paras.m = 0.9;
end

if ~isfield(paras,'GPU')
    paras.GPU = 0;
end

if ~isfield(paras,'loadmodel')
    doLOAD = 0;
    info.trainLoss = [];
    info.trainAcc = [];
    E0 = 0; % start from epoch 1
else
    doLOAD = 1;
    fprintf('loading saved epoch\n'); % the saved epoch include model, loss and accuracy
    load(paras.loadmodel);
    findEpochNum = find(paras.loadmodel(1:end-4)=='_',1,'last'); % saved epoch file name must be in the form of *_epoch number.mat
    E0 = str2double(paras.loadmodel(findEpochNum+1:end-4));
end

if ~isfield(paras,'saveprefix')
    doSAVE = 0;
else
    doSAVE = 1;
    if ~isfield(paras,'snapshot')
        paras.snapshot = paras.E; % only save the last epoch
    end
end



if isfield(paras,'testdat') && isfield(paras,'testlabel')
    doTEST = 1;
    testSiz = numel(paras.testlabel);
    info.testAcc = [];
    info.testLoss = [];
    if size(paras.testlabel,2)>1
        paras.testlabel = paras.testlabel';
    end
    if ~isfield(paras,'teststep')
        paras.teststep = 1;
    end
else
    doTEST = 0;
    info = [];
    return;
end


% Initalize momentum, learning rate multiplier, weightDecay multiplier, etc.
if doLOAD==0
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            for j=1:J
                net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end

if paras.GPU
  net = vl_simplenn_dcf_move(net, 'gpu') ;
  if doTEST
        paras.testdat = gpuArray(paras.testdat);
  end
end


% some global constant
if paras.GPU
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end


% validation
if doTEST
    infer_testLabel = [];
    info.testLoss(end+1)=0;
    for testB = 1:paras.B:testSiz
        
        fprintf('%4d',fix(testB/paras.B)+1);
        
        id = testB:min(testB+paras.B-1, testSiz);
        net.layers{end}.class = paras.testlabel(id) ;
        
        res_test = vl_simplenn7(net,paras.testdat(:,:,:,id),[],[], ...
             'conserveMemory', false, 'sync', true, ...
             'useMoments',true,'useDropout',false);
        
        [~,infer_testLabel(:,:,:,id)] = max(gather(res_test(end-1).x),[],3);
        info.testLoss(end)=info.testLoss(end)+gather(double(res_test(end).x));
        
        fprintf('\b\b\b\b');
        clear res_test;
        
    end
    
    infer_testLabel = squeeze(infer_testLabel);
    info.testAcc(end+1) = sum(infer_testLabel==paras.testlabel)/testSiz;
    info.testLoss(end) = info.testLoss(end)/testSiz;
    fprintf('  test loss: %.4f  test acc: %.2f%%\n', info.testLoss(end), 100*info.testAcc(end));
    clear res_test;
else
    fprintf('\n');
end


if doSAVE && rem(e-E0,paras.snapshot)==0
    save([paras.saveprefix,'_',num2str(e),'.mat'],'info','net');
end

