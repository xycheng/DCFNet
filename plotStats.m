function plotStats(stats,paras)

if ~isfield(paras,'expDir')
    paras.expDir = './output/';
end

%
modelFigPath = fullfile(paras.expDir, 'net-train.pdf') ;
epoch = numel(stats.train);
testt = paras.teststep;

%%
switchFigure(1) ; clf ;
plots = setdiff(...
    cat(2,...
    fieldnames(stats.train)', ...
    fieldnames(stats.val)'), {'num', 'time'}) ;
for p = plots
    p = char(p) ;
    values = zeros(0, epoch) ;
    leg = {} ;
    
    %
    f = char('train');
    tmp = [stats.(f).(p)];
    ind_train = (1:epoch)';
    values(end+1,ind_train) = tmp(1,:)';
    leg{end+1} =f;
    
    %
    f = char('val');
    tmp = [stats.(f).(p)];
    
    ind_val = (testt:testt:epoch)';
    values(end+1,ind_val) = tmp(1,:)';
    leg{end+1} =f;
    
%     
%     for f = {'train', 'val'}
%         f = char(f) ;
%         if isfield(stats.(f), p)
%             tmp = [stats.(f).(p)] ;
%             values(end+1,:) = tmp(1,:)' ;
%             leg{end+1} = f ;
%         end
%     end
    subplot(1,numel(plots),find(strcmp(p,plots))) ;
    
    %plot(1:epoch, values','o-') ;
    hold on;
    plot(ind_train, values(1,ind_train)','o-') ;
    plot(ind_val, values(2,ind_val)','o-') ;
    
    xlabel('epoch') ;
    title(p) ;
    legend(leg{:}) ;
    grid on ;
end
drawnow ;
print(1, modelFigPath, '-dpdf') ;

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