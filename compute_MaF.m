% Compute Multihop Assortativity Features for a set of graphs
% Author: Leonardo Gutierrez Gomez - leonardo.gutierrez@uclouvain.be
% Copyright 2018 - Leonardo Gutierrez Gomez
% Input  
% name:         string. Name of graph dataset in datasets folder
% nodel:        bynary. Whether nodes have labels (1) or not (0)
% l:            int. Time lag (hops) for the ofservable random process. default 3.
% num_eig:      int. Number of eigenvectors to compute. default 3.

% Output
% data_feats    a NxK matrix in which each row corresponds to a K
%               dimensional feature vector
%
% The output is saved as a dataframe in the features/ folder


% Example:      compute_MaF('mutag',1,3,3)


function [data_feats] = compute_MaF(name,nodel,l,num_eig)
tic;
py.importlib.import_module('networkx');

dataset = load(strcat('datasets/',name,'.mat'));
names   =  fieldnames(dataset);

labels  = getfield(dataset,names{1});
Graphs = getfield(dataset,names{2});

total = size(Graphs,2);
if nodel==1
  keys = node_labels(Graphs);
  vals = 1:size(keys,2);
  nodelabs = containers.Map(keys,vals); % mapping node categorical labels
end

for i=1:total
    
    N = size(Graphs(i).am,1);
    
    if N==2
        i=i+1
        N = size(Graphs(i).am,1);
    end
    A = Graphs(i).am;
    if issparse(A)
        A = full(A);
    end
    
    one = ones(N,1);
        
    G = create_graph(A);
    d = A*one;
    D = diag(d);
    m = double(G.number_of_edges());  
    
    M = inv(D)*A;
    pi = d'/(2*m); % pagerank
    display(i)
   
    lag = l;        
    H = eye(N);
    [Cov, Aut] = covariance(M,pi,H,lag);  

    [V D]=eigs(M',num_eig,'LM');

    
    pi2 = V(:,2);
    h2 =  pi2/norm(pi2,1); % second eigvector
    
    pi3 = V(:,3);
    h3 =  pi3/norm(pi3,1); % third eigenvector
    
    res1 = stability(Aut);
    res2 = attribute_covariance(Cov,pi',lag); % pagerank
    res3 = attribute_covariance(Cov,h2,lag);  % second eigenvector
    res4 = attribute_covariance(Cov,h3,lag); % third eivenvector
    
    if nodel==1
        H_labels   = compute_H(Graphs(i).nl.values,nodelabs);       
        [Cov, Aut] = covariance(M,pi,H_labels,lag);
        res5       = get_diagonal(Aut); %diag of Aut             
        r5(i,:)    = res5;              % Categorical node labels diag(Aut)
        
        r6(i,:)    = average_feats(H_labels,pi);
 
    end
           
    % inverse pi
    inv_pi = 1./pi';
       
    r1(i,:) = res1;       % H=identity
    r2(i,:) = real(res2); % pagerank
    r3(i,:) = real(res3); % second eigenvector of M    
    r4(i,:) = real(res4); % third eigenvector of M    
   
    
    r7(i,:) = m; % Number of edges
    
    r8(i,:) = pi*pi'; % avg pi
    r9(i,:) = pi*inv_pi; % avg inv_pi (num nodes)
    r10(i,:) = real(pi*pi2);  % avg second eigenvector
    r11(i,:) = real(pi*pi3); % avg third eigenvector
    
      
end;
if nodel==1
   feats     =  struct('f1',r1,'f2',r2,'f3',r3,'f4',r4,'f5',r5,'f6',r6,'f7',r7,'f8',r8,'f9',r9,'f10',r10,'f11',r11);
   res       =  [feats.f1,feats.f2,feats.f3,feats.f4,feats.f5,feats.f6,feats.f7,feats.f8, feats.f9,feats.f10,feats.f11, labels];
   sel_feats =  [1 2 3 4 5 6 7 8 9 10 11];
else
   feats     =  struct('f1',r1,'f2',r2,'f3',r3,'f4',r4,'f7',r7,'f8',r8,'f9',r9,'f10',r10,'f11',r11);
   res       =  [feats.f1,feats.f2,feats.f3,feats.f4,feats.f7,feats.f8, feats.f9,feats.f10,feats.f11, labels];
   sel_feats =  [1 2 3 4 7 8 9 10 11];
end

feats_table = create_dataframe(feats, sel_feats, res);

writetable(feats_table,strcat('features/',name,'_features.csv'));


toc

end

function [G] = create_graph(A)
    G = py.networkx.Graph;
    for i=1:size(A,1)
        for j=1:size(A,1)
            if A(i,j)>=1
                G.add_edge(i,j);
            end
        end
    end

end

function [Cov, Aut] = covariance(M,pi, H,lag)

    PI = diag(pi);
    
    for t = 0:lag
        C = PI*M^t - pi'*pi;
        R = H'*C*H;
        Cov{t+1} = C;
        Aut{t+1} = R;
    end    

end

function [stab] = stability(Aut)

    for j=1:size(Aut,2)
       stab(j) = trace(Aut{j}); 
    end
end


function [stab] = get_diagonal(Aut)
    K = size(Aut,2);
    
    stab = [];
    
    for k=1:K
        aCov = Aut{k};        
        stab = [stab diag(aCov)'];            

    end
    
end

function [u] = attribute_covariance(Cov,v,lag)
    u = zeros(1,lag+1);
    for t = 0:lag
        u(t+1) = v'*Cov{t+1}*v;
    end    

end

function [avg_feats] = average_feats(H,pi)

    for i=1:size(H,2)
        avg_feats(i) = pi*H(:,i);
        
    end

end

% Computing H equal dimension for all graphs
function [H] = compute_H(attribs, map_nodes_labs) 
    num_nodes = size(attribs,1);
    keys = map_nodes_labs;
    num_classes = size(keys,1);
    map = map_nodes_labs;
    
    H = zeros(num_nodes,num_classes);
    for i=1:num_nodes      
        H(i,map(attribs(i))) = 1;
    end;    
end

function [nodelabs] = node_labels(Graphs)
    nodelabs = []
    for i=1:size(Graphs,2)
       vals = Graphs(i).nl.values;
       res = unique(vals,'stable')';
       nodelabs = [nodelabs res];
       
    end
    nodelabs = unique(nodelabs,'first');
end

function [tit] = create_titles(sel_feats,titles, feats)
    lag = size(feats.f1,2)-1;
        
    index = 1;
    for f=sel_feats %scalar attribs
        if (f == 7)  || (f == 8) || (f == 9) || (f == 10) || (f == 11) 
                tit{index} = strcat(titles{f},'_',num2str(0));
                index = index+1;
        elseif (f == 5) % there are node attributes
             for t=0:size(feats.f5,2)-1
                tit{index} = strcat(titles{f},'_',num2str(t));
                index = index+1;   
             end
        elseif (f == 6) % there are average of node attributes
             for t=0:size(feats.f6,2)-1
                tit{index} = strcat(titles{f},'_',num2str(t));
                index = index+1;   
             end     
        else
            for t=0:lag
                tit{index} = strcat(titles{f},'_',num2str(t));
                index = index+1;
            end
        end
        
    end
    tit{index} = 'labels';
end


function [feats_table] = create_dataframe(feats, sel_feats,res)
    
    tit1 = 'ID'; tit2 = 'pagerank'; tit3 = 'eigenvec_2nd_M'; 
    tit4 = 'eigenvec_3rd_M'; tit5 = 'node_attribs'; tit6 = 'avg_node_labels'; 
    tit7 = 'n_edges'; tit8 = 'avg_pi'; tit9 = 'avg_inv_pi'; 
    tit10 = 'avg_eigenvec_2nd'; tit11 = 'avg_eigenvec_3rd';
    
    
    titles = {tit1,tit2,tit3,tit4,tit5,tit6,tit7,tit8,tit9,tit10,tit11};

    header = create_titles(sel_feats,titles, feats);
    feats_table = array2table(res);
    
    feats_table.Properties.VariableNames=header;
    
        
end
