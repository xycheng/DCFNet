function net = vl_simplenn_dcf_move(net, destination)
% modifed from VL_SIMPLENN_MOVE  in matconvnet package

switch destination
    
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end

for l=1:numel(net.layers)

    switch net.layers{l}.type
        
        case {'conv', 'conv-dcf','conv-psi',...
                'convt',...
                'bnorm'}
            
            for f = {'filters', 'biases','filtersMomentum', 'biasesMomentum'}
                f = char(f) ;
                if isfield(net.layers{l}, f)
                    net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
                end
            end
            
            for f = {'weights', 'psiweights',...
                    'bases',...
                    'momentum'}
                f = char(f) ;
                if isfield(net.layers{l}, f)
                    for j=1:numel(net.layers{l}.(f))
                        net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
                    end
                end
            end
            
        otherwise
            % nothing to do ?
    end
end
