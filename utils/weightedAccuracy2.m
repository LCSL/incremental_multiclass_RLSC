function [wAcc , CM] = weightedAccuracy2( Y, Ypred , classes)
%WEIGHTEDACCURACY works for +-1 coding
%   Detailed explanation goes here

    [~,Yvec] = max(Y,[],2);
    [~,Ypredvec] = max(Ypred,[],2);


    CM = confusionmat(Yvec,Ypredvec);
    seenClasses = union(Yvec,Ypredvec);
    sd = setdiff(1:numel(classes), seenClasses);
    if numel(sd) > 0
        for i = 1:numel(sd)
            CM = [CM(:,1:sd(i)-1) , zeros(size(CM,1),1), CM(:,sd(i):size(CM,2)) ];    % Add 0 column
            CM = [CM(1:sd(i)-1,:) ; zeros(1,size(CM,2)); CM(sd(i):size(CM,1),:) ];    % Add 0 row                        
        end
    end
    CM(seenClasses,seenClasses) = ...
    CM(seenClasses,seenClasses) ./ repmat(max(sum(CM(seenClasses,seenClasses),2),1),1,numel(seenClasses));
    wAcc = trace(CM)/numel(seenClasses);
end
