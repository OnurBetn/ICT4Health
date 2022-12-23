clear; clc; close all
%%
rng('default');
tstart=tic;
Kquant=9; % number of quantization levels
Nstates=8; % number of states in the HMM
% Subset 1
ktrain=[1,2,3,4,5,6,7]; % indexes of patients for training
ktest=[8,9,10]; % indexes of patients for testing
% % Subset 2
% ktrain=[4,5,6,7,8,9,10]; % indexes of patients for training
% ktest=[1,2,3]; % indexes of patients for testing
% % Subset 3
% ktrain=[1,2,3,4,8,9,10]; % indexes of patients for training
% ktest=[5,6,7]; % indexes of patients for testing
[hq,pq]=pre_process_data(Nstates,Kquant,ktrain); % generate the quantized signals
telapsed = toc(tstart);
disp(['first part, elapsed time ',num2str(telapsed),' s'])

%% HMM training phase....
tstart=tic;
% 1st option for transition matrix
p = 0.9;
q = (1-p)/(Nstates-1);
TR_HAT = gallery('circul', [q p repmat(q,1,Nstates-2)]); % circulant matrix with first row [q,p,q,q,...,q]

% 2nd option for transition matrix
% TR_HAT = rand(Nstates, Nstates);
% rowsum = sum(TR_HAT,2);
% TR_HAT = TR_HAT./rowsum; % each row has to sum up to 1 (stochastic matrix)

% Emission matrix
EMIT_HAT = rand(Nstates, Kquant);
rowsum = sum(EMIT_HAT,2);
EMIT_HAT = EMIT_HAT./rowsum; % each row has to sum up to 1 (stochastic matrix)

% Training HC-HMM
[TR_HC,EMIT_HC]=hmmtrain(hq(ktrain),TR_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);
% Training PD-HMM
[TR_PD,EMIT_PD]=hmmtrain(pq(ktrain),TR_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);

telapsed = toc(tstart);
disp(['training part, elapsed time ',num2str(telapsed),' s'])

%% HMM testing phase....
train_set = [hq(ktrain), pq(ktrain)];
test_set = [hq(ktest), pq(ktest)];

% in y vectors 0 stands for HC, 1 for PD
y_train = [zeros(size(ktrain,2),1); ones(size(ktrain,2),1)];
y_test = [zeros(size(ktest,2),1); ones(size(ktest,2),1)];

y_hat_train = zeros(size(train_set,2),1);
for i=1:size(train_set,2)
    [~,LOGP_HC] = hmmdecode(train_set{1,i},TR_HC,EMIT_HC);
    [~,LOGP_PD] = hmmdecode(train_set{1,i},TR_PD,EMIT_PD);
    if LOGP_PD > LOGP_HC
        y_hat_train(i) = 1;
    end
end

yy = or(y_hat_train, y_train); % true negatives if yy==0 (if y and y_hat are both 0)
train_spec = sum(yy == 0) / sum(y_train == 0); % = TN / (FP + TN)
disp(['training specificity: ',num2str(train_spec)])

yy = and(y_hat_train, y_train); % true positives if yy==1 (if y and y_hat are both 1)
train_sens = sum(yy == 1) / sum(y_train == 1); % = TP / (TP + FN)
disp(['training sensitivity: ',num2str(train_sens)])

y_hat_test = zeros(size(test_set,2),1);
for i=1:size(test_set,2)
    [~,LOGP_HC] = hmmdecode(test_set{1,i},TR_HC,EMIT_HC);
    [~,LOGP_PD] = hmmdecode(test_set{1,i},TR_PD,EMIT_PD);
    if LOGP_PD > LOGP_HC
        y_hat_test(i) = 1;
    end
end

yy = or(y_hat_test, y_test); % true negatives if yy==0 (if y and y_hat are both 0)
test_spec = sum(yy == 0) / sum(y_test == 0); % = TN / (FP + TN)
disp(['test specificity: ',num2str(test_spec)])

yy = and(y_hat_test, y_test); % true positives if yy==1 (if y and y_hat are both 1)
test_sens = sum(yy == 1) / sum(y_test == 1); % = TP / (TP + FN)
disp(['test sensitivity: ',num2str(test_sens)])

