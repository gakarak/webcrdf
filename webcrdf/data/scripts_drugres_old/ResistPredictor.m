classdef ResistPredictor
    properties
        props = {}; % properties and features of the input data tables
%         jj = [];  % field: active (non-zero) columns
%         mn;       % field: mean values of columns
%         pc;       % field: PC matrix
%         chosen;   % field: number(s) of chosen PC(s)
        
        nmax = [];  % number of PC's to use from each table
        
        svmlin;     % trained classifiers
        svmquad;
        logit;
        
        trained_on = 0; % number of cases the classifiers trained on
    end
    
    methods(Static = true)       
        function obj = fit(tbls, suppl, DR, max_pc)
            obj = ResistPredictor;
            
            if length(unique(DR)) > 2
                DR = double(DR > min(DR));
            end
            
            if ~iscell(tbls)
                tbls = {tbls};
            end
            
            if nargin < 4
                obj.nmax = ones(1, length(tbls));
            else
                obj.nmax = max_pc;
                if length(obj.nmax) == 1
                    obj.nmax = ones(1, length(tbls)) * obj.nmax;
                end
                if length(obj.nmax) ~= length(tbls)
                    throw(MException('VerifyInput:MaxPC', ...
                        'Wrong max_pc parameter length'));
                end
            end
            nmx = obj.nmax;
          
            X = [];
            obj.props = cell(1, length(tbls));
            
            for it = 1:length(tbls)
                tbl = tbls{it};
                pr.jj = max(tbl) > min(tbl);
                pr.mn = mean(tbl);
                [pr.pc, sc] = princomp(tbl(:, pr.jj), 'econ');
                fact = sc(:, 1:12);
                [rs, ~] = corr(fact, DR);
                [~, idx] = sort(abs(rs), 'descend');
                pr.chosen = idx(1:nmx(it));
                
                X = [X sc(:, pr.chosen)];
                obj.props{it} = pr;
            end
            
            if ~isempty(suppl)
                X = [X suppl];
            end
            
            obj.svmlin = svmtrain(X, DR, ...
                'Kernel_Function', 'Linear');
            obj.svmquad = svmtrain(X, DR, ...
                'Kernel_Function', 'Quadratic', ...
                'options', optimset('MaxIter', 3e5));
            obj.logit = mnrfit(X, DR + 1);
            
            obj.trained_on = length(DR);
        end
    end
    
    methods(Static = false)
        function [svml_c, svmq_c, logit_c, logit_prob] = ...
                predict(obj, tbls, suppl)
            
            if ~iscell(tbls)
                tbls = {tbls};
            end
            
            if length(tbls) ~= length(obj.props)
                throw(MException('VerifyInput:NumberOfTables', ...
                    sprintf('Number of input tablers should be %i', ...
                    length(obj.props))));
            end
            
            X = [];            
            for it = 1:length(tbls)
                tbl = tbls{it};
                tbl = scal(tbl, obj.props{it}.mn, ones(1, size(tbl, 2)));
                sc = tbl(:, obj.props{it}.jj) * obj.props{it}.pc;
                X = [X sc(:, obj.props{it}.chosen)];
            end
            
            if ~isempty(suppl)
                X = [X suppl];
            end
            
            svml_c = svmclassify(obj.svmlin, X);
            svmq_c = svmclassify(obj.svmquad, X);
            
            logit_prob = mnrval(obj.logit, X);
            logit_prob = logit_prob(:, 2);
            logit_c = logit_prob > 0.5;
        end
    end
end