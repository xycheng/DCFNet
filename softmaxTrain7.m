function [net, info] = softmaxTrain7(trainX,trainLabel,net,paras)

% changed from CNN_MNIST  Demonstrates MatConvNet on MNIST

%%
imdb.images = [];
imdb.images.data = cat(4, trainX, paras.testdat);
imdb.images.labels = cat(2, trainLabel, paras.testlabel);
imdb.images.set = [ones(1,numel(trainLabel)), 3*ones(1,numel(paras.testlabel))];

%%
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


if ~isfield(paras,'expDir')
    paras.expDir = './output/';
end

% if ~exist(paras.expDir)
%     mkdir(paras.expDir);
% end

if ~isfield(paras,'snapshot')
    paras.snapshot = paras.E; % only save the last epoch
end
    

if ~isfield(paras,'teststep')
    paras.teststep = 1; %val in each epoch
end


%%
opts.train=[];
opts.train.batchSize = paras.B;
opts.train.numEpochs = paras.E;
opts.train.learningRate = paras.eta;
opts.train.weightDecay =  paras.w ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

%%

opts.expDir = paras.expDir;

opts.continue = true ;
opts.batchSize = paras.B;
opts.numSubBatches = 1 ;

opts.train = [] ;
opts.val =  find(imdb.images.set == 3);

opts.gpus = []; 

if paras.GPU > 0
    opts.gpus = paras.GPU;
end

opts.epochSize = inf;
opts.prefetch = false ;
opts.numEpochs = paras.E ;

opts.learningRate = paras.eta;
opts.weightDecay = paras.w;

opts.solver = [] ;  % Empty array means use the default SGD solver
%[opts, varargin] = vl_argparse(opts, varargin) ;

if ~isempty(opts.solver)
  assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
    'Invalid solver; expected a function handle with two outputs.') ;
  % Call without input arguments, to get default options
  opts.solverOpts = opts.solver() ;
end

opts.momentum = 0.9 ;
opts.saveSolverState = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;

opts.plotStatistics = false;

opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
%opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train)
  opts.train = [] ;
end
if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val)
  opts.val = [] ;
end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

net = vl_simplenn_dcf_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  for i=1:numel(net.layers)
    J = numel(net.layers{i}.weights) ;
    if ~isfield(net.layers{i}, 'learningRate')
      net.layers{i}.learningRate = ones(1, J) ;
    end
    if ~isfield(net.layers{i}, 'weightDecay')
      net.layers{i}.weightDecay = ones(1, J) ;
    end
  end
end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
      hasError = false ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
    otherwise
      error('Unknown error function ''%s''.', opts.errorFunction) ;
  end
end

state.getBatch = getBatch ;
stats = [] ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
end

for epoch=start+1:opts.numEpochs

  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  % Train for one epoch.
  params = opts ;
  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  params.train = params.train(1:min(opts.epochSize, numel(opts.train)));
  params.val = opts.val(randperm(numel(opts.val))) ;
  params.imdb = imdb ;
  params.getBatch = getBatch ;
  
  %%%
  valThisStep = 0;
  if rem(epoch, paras.teststep)==0 
      valThisStep =1;
  end
  
  %%%
  

  if numel(params.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train') ;
    
    if valThisStep
        [net, state] = processEpoch(net, state, params, 'val') ;
    end
        
    if ~evaluateMode
        
        if rem(epoch,paras.snapshot)==0
            saveState(modelPath(epoch), net, state) ;
        end
    end
    
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, state, params, 'train') ;
      
      if valThisStep
          [net, state] = processEpoch(net, state, params, 'val') ;
          
      end
      
      if labindex == 1 && ~evaluateMode
          
          if rem(epoch,paras.snapshot)==0
              saveState(modelPath(epoch), net, state) ;
          end
      end

      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(epoch) = lastStats.train ;
  
  if valThisStep 
      stats.val(epoch) = lastStats.val ;
  end
  
  clear lastStats ;
  
  if ~evaluateMode
      
      if rem(epoch,paras.snapshot)==0
          saveStats(modelPath(epoch), stats) ;
      end
  end

  if params.plotStatistics
  %if params.plotStatistics || valThisStep
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
  
  if ~isempty(opts.postEpochFn)
    if nargout(opts.postEpochFn) == 0
      opts.postEpochFn(net, params, state) ;
    else
      lr = opts.postEpochFn(net, params, state) ;
      if ~isempty(lr), opts.learningRate = lr; end
      if opts.learningRate == 0, break; end
    end
  end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end



%%
info = stats;




function fn = getBatch()
% --------------------------------------------------------------------
fn = @(x,y) getSimpleNNBatch(x,y) ;


% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% -------------------------------------------------------------------------
function err = error_multiclass(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binary(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(params, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
  for i = 1:numel(net.layers)
    state.solverState{i} = cell(1, numel(net.layers{i}.weights)) ;
    state.solverState{i}(:) = {0} ;
  end
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net = vl_simplenn_dcf_move(net, 'gpu') ;
  for i = 1:numel(state.solverState)
    for j = 1:numel(state.solverState{i})
      s = state.solverState{i}{j} ;
      if isnumeric(s)
        state.solverState{i}{j} = gpuArray(s) ;
      elseif isstruct(s)
        state.solverState{i}{j} = structfun(@gpuArray, s, 'UniformOutput', false) ;
      end
    end
  end
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  vl_simplenn_start_parserv(net, parserv) ;
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

subset = params.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;

start = tic ;

prlen1 = 0;prlen2 = 0;prlen3 = 0;

for t=1:params.batchSize:numel(subset)
  
  fprintf(repmat('\b',[1,prlen1+prlen2+prlen3]));
  prlen1=fprintf('%s: epoch %02d: %3d/%3d:', mode, params.epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
      
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    [im, labels] = params.getBatch(params.imdb, batch) ;

    if params.prefetch
      if s == params.numSubBatches
        batchStart = t + (labindex-1) + params.batchSize ;
        batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
      params.getBatch(params.imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    if strcmp(mode, 'train')
      dzdy = 1 ;
      useMoments = false;
      useDropout = true;
      
    else
      dzdy = [] ;
      useMoments = true;
      useDropout = false; 
    end
    
    net.layers{end}.class = labels ;
    
    
    
    res = vl_simplenn7(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'useMoments', useMoments, 'useDropout', useDropout, ...
                      'conserveMemory', params.conserveMemory, ...
                      'backPropDepth', params.backPropDepth, ...
                      'sync', params.sync, ...
                      'cudnn', params.cudnn, ...
                      'parameterServer', parserv, ...
                      'holdOn', s < params.numSubBatches) ;
                      
                  

    % accumulate errors
    error = sum([error, [...
      sum(double(gather(res(end).x))) ;
      reshape(params.errorFunction(params, labels, res),[],1) ; ]],2) ;
  end

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv) ;
  end

  % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = extractStats(net, params, error / num) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1
    % compensate for the first three iterations, which are outliers
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  prlen2=fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  
   prlen3 = 0;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    prlen3=prlen3+fprintf(' %s: %.3f', f, stats.(f)) ;
  end
  
  %fprintf('\n') ;

  % collect diagnostic statistics
  if strcmp(mode, 'train') && params.plotDiagnostics
    switchFigure(2) ; clf ;
    diagn = [res.stats] ;
    diagnvar = horzcat(diagn.variation) ;
    diagnpow = horzcat(diagn.power) ;
    subplot(2,2,1) ; barh(diagnvar) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTick', 1:numel(diagnvar), ...
      'YTickLabel',horzcat(diagn.label), ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1], ...
      'XTick', 10.^(-5:1)) ;
    grid on ; title('Variation');
    subplot(2,2,2) ; barh(sqrt(diagnpow)) ;
    set(gca,'TickLabelInterpreter', 'none', ...
      'YTick', 1:numel(diagnpow), ...
      'YTickLabel',{diagn.powerLabel}, ...
      'YDir', 'reverse', ...
      'XScale', 'log', ...
      'XLim', [1e-5 1e5], ...
      'XTick', 10.^(-5:5)) ;
    grid on ; title('Power');
    subplot(2,2,3); plot(squeeze(res(end-1).x)) ;
    drawnow ;
  end
end

fprintf('\n')

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveSolverState
  state.solverState = [] ;
else
  for i = 1:numel(state.solverState)
    for j = 1:numel(state.solverState{i})
      s = state.solverState{i}{j} ;
      if isnumeric(s)
        state.solverState{i}{j} = gather(s) ;
      elseif isstruct(s)
        state.solverState{i}{j} = structfun(@gather, s, 'UniformOutput', false) ;
      end
    end
  end
end

net = vl_simplenn_dcf_move(net, 'cpu') ;

% -------------------------------------------------------------------------
function [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for l=numel(net.layers):-1:1
  for j=numel(res(l).dzdw):-1:1

    if ~isempty(parserv)
      tag = sprintf('l%d_%d',l,j) ;
      parDer = parserv.pull(tag) ;
    else
      parDer = res(l).dzdw{j}  ;
    end

    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = vl_taccum(...
        1 - thisLR, ...
        net.layers{l}.weights{j}, ...
        thisLR / batchSize, ...
        parDer) ;
    else
      % Standard gradient training.
      thisDecay = params.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = params.learningRate * net.layers{l}.learningRate(j) ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.layers{l}.weights{j}) ;

        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{l}{j} = vl_taccum(...
            params.momentum, state.solverState{l}{j}, ...
            -1, parDer) ;

          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{l}{j} - parDer ;
          else
            delta = state.solverState{l}{j} ;
          end

          % Update parameters.
          net.layers{l}.weights{j} = vl_taccum(...
            1, net.layers{l}.weights{j}, ...
            thisLR, delta) ;

        else
          % call solver function to update weights
          [net.layers{l}.weights{j}, state.solverState{l}{j}] = ...
            params.solver(net.layers{l}.weights{j}, state.solverState{l}{j}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    end

    % if requested, collect some useful stats for debugging
    if params.plotDiagnostics
      variation = [] ;
      label = '' ;
      switch net.layers{l}.type
        case {'conv','convt'}
          if isnumeric(state.solverState{l}{j})
            variation = thisLR * mean(abs(state.solverState{l}{j}(:))) ;
          end
          power = mean(res(l+1).x(:).^2) ;
          if j == 1 % fiters
            base = mean(net.layers{l}.weights{j}(:).^2) ;
            label = 'filters' ;
          else % biases
            base = sqrt(power) ;%mean(abs(res(l+1).x(:))) ;
            label = 'biases' ;
          end
          variation = variation / base ;
          label = sprintf('%s_%s', net.layers{l}.name, label) ;
      end
      res(l).stats.variation(j) = variation ;
      res(l).stats.power = power ;
      res(l).stats.powerLabel = net.layers{l}.name ;
      res(l).stats.label{j} = label ;
    end
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net, params, errors)
% -------------------------------------------------------------------------
stats.objective = errors(1) ;
for i = 1:numel(params.errorLabels)
  stats.(params.errorLabels{i}) = errors(i+1) ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, state)
% -------------------------------------------------------------------------
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = vl_simplenn_dcf_tidy(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
%clear vl_tmove vl_imreadjpeg ;
disp('Clearing mex files') ;
clear mex ;
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(params, cold)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end
end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename) ;
  clearMex() ;
  if numGpus == 1
    disp(gpuDevice(params.gpus)) ;
  else
    spmd
      clearMex() ;
      disp(gpuDevice(params.gpus(labindex))) ;
    end
  end
end


