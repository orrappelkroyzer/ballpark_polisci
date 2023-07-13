# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:57:11 2020

@author: tomhope
"""

# imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import pickle
import errno
from random import shuffle
from ballpark_yhat_reg import solve_w_y_dccp,feasibility_regression,solve_w_y
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from Bags import BagExtractor
import argparse

import io

def evaluate(X, data, w, y, info_string,file_path,num_to_save=125, clf=None):
    """
      This function creates top,mid,bottom results by ranking from ballpark model
      These resutls are are saved to create the html display
    
    Args:
        X (numpy): features (of the same dim and scaling used to train model weights w)
        data (pandas): patent metadata, where each row corresponds to row in X
        w (numpy): vector of model weights
        y (numpy): vector of model latent labels (or scores)
        info_string (str): info string for names of saved files
        file_path (Path) path of folder to save the results
        num_to_save(int): Number of results to save
        
    Returns:
        saves y and score to disk, under file_path

    """
    if "svm_news" in info_string:
        score = clf.decision_function(X)
    elif "svm_citation" in info_string:
        score = clf.predict(X).squeeze()
    else:
        score = np.dot(X,w)

    print("average score: " + str(np.mean(score)))
    print("number of ranked patents: " + str(score.shape[0]))
    print("number of patents to save: "+ str(num_to_save))

    rank = []
    score_rank = []
    y_rank = []
    for i in range(len(score)):
        title = data.iloc[i].title
        pid = data.iloc[i].name
        rank.append((title, score[i], pid, i, y[i]))
        score_rank.append((score[i],pid))
        y_rank.append((y[i],pid))
    sorted_s = sorted(score_rank, key=itemgetter(0))
    sorted_y = sorted(y_rank, key=itemgetter(0))
    mid = int(len(rank)*0.5)
    
#    plt.hist(scores,100)
#    np.percentile(scores,99)
#    
    res = [list(reversed(sorted_s[-num_to_save:])), list(reversed(sorted_s[mid-num_to_save:mid])), list(reversed(sorted_s[:num_to_save]))]
    res_y = [list(reversed(sorted_y[-num_to_save:])), list(reversed(sorted_y[mid-num_to_save:mid])), list(reversed(sorted_y[:num_to_save]))]

    score_file = 'ballpark_results'+str(info_string)+'_score.pickle'
    with open(str(file_path)+"\\"+score_file, 'wb') as handle:
            pickle.dump(res, handle)
            
    y_file = 'ballpark_results'+str(info_string)+'_y.pickle'
    with open(str(file_path)+"\\"+y_file, 'wb') as handle:
            pickle.dump(res_y, handle)


def generate_results_html(name, results,data):
    """
      This function creates HTMLs for annotation by experts
      Each HTML file contains top,mid,bottom results (ranked by model) --> randomized
      Each HTML folder contains a shuffle file indicating the random->original mapping of conditions
      
    Args:
        name (Path): path of folder to save the HTML
        results (list): list of results created with evaluate() function
        data (pandas): patent metadata, with index = patendID
        
    Returns:
        saves HTMLs , under name

    """


    URL_FORMAT = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm&r=1&f=G&l=50&s1={0}.PN.&OS=PN/{0}&RS=PN/{0}'
    HTML = '<p>{1})<a href="'+ URL_FORMAT +'">{2}</a></p><p>assignee: {3}, year: {4}</p>'
    HTML_HIDE_ABS = '<style>div {display: none;}</style>'
    HTML_BUTTON = '<button onclick="myFunction(\'myDIV{0}\')">abs</button> <div id="myDIV{0}">{1}</div>'
    HTML_ABSTRACT_BUTTON = '<script> function myFunction(elem) {var x = document.getElementById(elem); ' \
                           'if (x.style.display != "block") {x.style.display = "block";} ' \
                           'else {x.style.display = "none";}}</script>'

    for i in range(len(results)):
        shuffle(results[i])

    index_key = {0: [], 1: [], 2: [], 3: [], 4: []}
    if not name.exists():
        try:
            name.mkdir(parents=True)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    for i in range(5):
        order = list(range(3))
        shuffle(order)
        res_name = 'results'+ str(i) +'.html'
        res_name = name/res_name
        #print(res_name)
        with io.open(str(res_name),'w',encoding="utf-8") as h:
            #print("writing ", res_name)
            h.write(HTML_HIDE_ABS)
            for j in range(3):
                index_key[i].append((j, order[j]))
                counter = 0
                h.write('<h2>--------------------group'+ str(j) +'----------------------</h2>')
                for score,pid in results[order[j]][10*i:10*i+10]:
                    patdat = data.loc[pid]
                    year = patdat.app_date
                    assignee = patdat.company
                    counter += 1
                    h.write(HTML.format(pid, counter, patdat.title, assignee, year) )
                    h.write(HTML_BUTTON.format(counter + j*10, patdat.abstract) )
            h.write(HTML_ABSTRACT_BUTTON )
            
    with io.open(str(name/'shuffle_index.txt'),'w',encoding="utf-8") as f:
        #print("writing ", name/'shuffle_index.txt' )
        for key in index_key:
            f.write('------------------ file ' + str(key) + '----------------------\n' )
            for elem in index_key[key]:
                if elem[1] == 0:
                    f.write(str(elem[0])  + ' -> top\n' )
                elif elem[1] == 1:
                    f.write(str(elem[0])  + ' -> mid\n' )
                elif elem[1] == 2:
                    f.write(str(elem[0])  + ' -> bot\n' )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,default="C:\\data\\patents\\")
    parser.add_argument('--nlp_features_folder_train', type=str,default="output_features_patent_texts_train")
    parser.add_argument('--patent_features_train', type=str,default="ballpark_features_train.csv")
    parser.add_argument('--patent_metadata_train', type=str,default="patent_metadata_train.csv")
    parser.add_argument('--use_abstract_vector', action='store_true', help="with abstract vector,default false")
    parser.add_argument('--model_type', type=str,default="feasibility",choices=['svm_news','svm_citation','feasibility','latent_y', 'dccp','dccp_convex_init', 'anomaly_detection'])
    parser.add_argument('--lp_news', type=float, default=None)
    parser.add_argument('--lp_comp', type=float, default=None)
    parser.add_argument('--lp_uni', type=float, default=None)
    parser.add_argument('--lp_cities', type=float, default=None)
    parser.add_argument('--up_news', type=float, default=None)
    parser.add_argument('--up_comp', type=float, default=None)
    parser.add_argument('--up_uni', type=float, default=None)
    parser.add_argument('--up_cities', type=float, default=None)
    parser.add_argument('--lq_news', type=float, default=None)
    parser.add_argument('--lq_comp', type=float, default=None)
    parser.add_argument('--lq_uni', type=float, default=None)
    parser.add_argument('--lq_cities', type=float, default=None)
    parser.add_argument('--uq_news', type=float, default=None)
    parser.add_argument('--uq_comp', type=float, default=None)
    parser.add_argument('--uq_uni', type=float, default=None)
    parser.add_argument('--uq_cities', type=float, default=None)
    parser.add_argument('--l_pairs_news', type=float, default=None)
    parser.add_argument('--l_pairs_comp', type=float, default=None)
    parser.add_argument('--l_pairs_uni', type=float, default=None)
    parser.add_argument('--l_pairs_cities', type=float, default=None)
    parser.add_argument('--u_pairs_news', type=float, default=None)
    parser.add_argument('--u_pairs_comp', type=float, default=None)
    parser.add_argument('--u_pairs_uni', type=float, default=None)
    parser.add_argument('--u_pairs_cities', type=float, default=None)
    parser.add_argument('--lp_all_news', type=float, default=None)
    parser.add_argument('--lp_all_comp', type=float, default=None)
    parser.add_argument('--lp_all_uni', type=float, default=None)
    parser.add_argument('--lp_all_cities', type=float, default=None)
    parser.add_argument('--up_all_news', type=float, default=None)
    parser.add_argument('--up_all_comp', type=float, default=None)
    parser.add_argument('--up_all_uni', type=float, default=None)
    parser.add_argument('--up_all_cities', type=float, default=None)
    parser.add_argument('--lq_all_news', type=float, default=None)
    parser.add_argument('--uq_all_news', type=float, default=None)
    parser.add_argument('--lp_small_comp', type=float, default=None)
    parser.add_argument('--up_small_comp', type=float, default=None)
    parser.add_argument('--l_all', type=float, default=None)
    parser.add_argument('--u_all', type=float, default=None)
    parser.add_argument('--num_to_save', type=int, default=None)


    args = parser.parse_args()
    print(args)
    
    ##PATHS TO FEATURES, DATA, BAGS
    file_path = Path(args.file_path)
    nlp_features_folder_train = args.nlp_features_folder_train
    patent_features_train = args.patent_features_train
    patent_metadata_train = args.patent_metadata_train

    l_pairs = {'news': args.l_pairs_news, 'comp': args.l_pairs_comp, 'uni': args.l_pairs_uni, 'cities': args.l_pairs_cities}
    u_pairs = {'news': args.u_pairs_news, 'comp': args.u_pairs_comp, 'uni': args.u_pairs_uni, 'cities': args.u_pairs_cities}
    l = {'news': {'p': args.lp_news, 'q': args.lq_news, 'p_all': args.lp_all_news, 'q_all': args.lq_all_news},
         'comp': {'p': args.lp_comp, 'q': args.lq_comp, 'p_all': args.lp_all_comp, 'p_small': args.lp_small_comp},
         'uni': {'p': args.lp_uni, 'q': args.lq_uni, 'p_all': args.lp_all_uni},
         'cities': {'p': args.lp_cities, 'q': args.lq_cities, 'p_all': args.lp_all_cities},
         'all': args.l_all}
    u = {'news': {'p': args.up_news, 'q': args.uq_news, 'p_all': args.up_all_news, 'q_all': args.uq_all_news},
         'comp': {'p': args.up_comp, 'q': args.uq_comp, 'p_all': args.up_all_comp,  'p_small': args.up_small_comp},
         'uni': {'p': args.up_uni, 'q': args.uq_uni, 'p_all': args.up_all_uni},
         'cities': {'p': args.up_cities, 'q': args.uq_cities, 'p_all': args.up_all_cities},
         'all': args.u_all}

    num_to_save = args.num_to_save

    nlp_features_folder = file_path/nlp_features_folder_train
    patent_features = file_path/patent_features_train
    patent_citations = file_path/"ballpark_citations_train.csv"

    patent_metadata =  file_path/patent_metadata_train
    
    #print(patent_metadata)
    #print(nlp_features_folder)
    #print(patent_features)
    
    bag_path = file_path/"News Bags"
    be = BagExtractor(str(bag_path))
    bags = be.get_bags_by_index()

    X = pd.read_csv(patent_features)
    data = pd.read_csv(patent_metadata,index_col="pid")
    cites = pd.read_csv(patent_citations)
    model_type = args.model_type
    #WHICH BP MODEL, WHICH FEATURES 
    use_abstract_vector = args.use_abstract_vector
    
    #X.drop(columns=['generality', 'originality'],inplace=True)
    
    
#    ###TODO CHECK THE WEIRD STUFF GOING ON WITH NEWS...
#    news_scrape_data = pd.read_csv(file_path/"patent_news_scrape_dict.csv",header=None)
#    rows_in_news = []
#    for t in news_scrape_data[1].values:
#         t=  ' '.join(t.split())
#         #t= t.capitalize()
#         if len(data[data.title==t]):
#             l = data.index.get_loc(data[data.title==t].index[0])
#             rows_in_news.append(l)
#             
#    diff_bags1 = [r for r in rows_in_news if r not in bags["News"]]    
#    diff_bags = [b for b in bags["News"] if b not in rows_in_news]
#    data.iloc[diff_bags]
#    ##############################
    
    if use_abstract_vector:
        use_abstract = "title_abstract"
    else:
        use_abstract = "title"
    
    if "dccp" in model_type:
        use_dccp = "dccp"
    else:
        use_dccp = ""
        
    if model_type == "dccp_convex_init":
        use_convex = "convex_init"
    else:
        use_convex =""
    
    files = [x for x in nlp_features_folder.iterdir() if x.is_file() and x.suffix==".csv"]
    if not use_abstract_vector:
        print("Using nlp features without abstract")
        files = [f for f in files if "wikibert_title_abstract" not in f.name]
    assert(len(files))
    
    nlp_features = []
    for f in files:
        features = pd.read_csv(f,header=None)
        nlp_features.append(features)
    
    X_features = pd.concat(nlp_features+[X],axis=1)    
    
    #SCALE FEATURES
    scaler = MinMaxScaler()
    X_features = scaler.fit_transform(X_features)
    
#    #CLEAN BAGS -- ONLY BAGS WITH MORE THAN ONE PATENT
#    good_bags = {}
#    for k,v in bags.items():
#        if len(v)>1:
#            good_bags[k] = v
#    

    
    #BUILD CONSTRAINTS
    #TODO -- DIFFERENT THRESHOLD FOR NEWS, COMP/UNI
#    good_bags["all"] = list(range(len(X_features)))
    pairwise_constraints = be.get_pairs()
    lower_diff_bound_bags = be.get_lower_bounds_of_pairs_diff(news_arg=l_pairs['news'],comp_arg = l_pairs['comp'],
                                                              uni_arg=l_pairs['uni'], cities_arg=l_pairs['cities'])
    upper_diff_bound_bags = be.get_upper_bounds_of_pairs_diff(news_arg=u_pairs['news'],comp_arg = u_pairs['comp'],
                                                              uni_arg=u_pairs['uni'], cities_arg=u_pairs['cities'])
    lower_p_bound_bags = be.get_lower_bounds_of_bags(
        p_news_arg=l['news']['p'], p_comp_arg=l['comp']['p'], p_uni_arg=l['uni']['p'], p_cities_arg=l['cities']['p'],
        q_news_arg=l['news']['q'], q_comp_arg=l['comp']['q'], q_uni_arg=l['uni']['q'], q_cities_arg=l['cities']['q'],
        p_all_news_arg=l['news']['p_all'], p_all_comp_arg=l['comp']['p_all'], p_all_uni_arg=l['uni']['p_all'], p_all_cities_arg=l['cities']['p_all'],
        q_all_news_arg=l['news']['q_all'], p_small_comp_arg=l['comp']['p_small'], p_all=l['all'])
    upper_p_bound_bags = be.get_upper_bounds_of_bags(
        p_news_arg=u['news']['p'], p_comp_arg=u['comp']['p'], p_uni_arg=u['uni']['p'], p_cities_arg=u['cities']['p'],
        q_news_arg=u['news']['q'], q_comp_arg=u['comp']['q'], q_uni_arg=u['uni']['q'], q_cities_arg=u['cities']['q'],
        p_all_news_arg=u['news']['p_all'], p_all_comp_arg=u['comp']['p_all'], p_all_uni_arg=u['uni']['p_all'], p_all_cities_arg=u['cities']['p_all'],
        q_all_news_arg=u['news']['q_all'], p_small_comp_arg=u['comp']['p_small'], p_all=u['all'])

    if 'comp_Rel_Not In Comps' in upper_p_bound_bags:
        del upper_p_bound_bags['comp_Rel_Not In Comps']
    if 'uni_Rel_Not In Uni' in upper_p_bound_bags:
        del upper_p_bound_bags['uni_Rel_Not In Uni']


    #upper_p_bound_bags["comp_All_Not In Comps"] = 0.05
    #upper_p_bound_bags["uni_Rel_Not In Uni"] = 0.05
#    upper_p_bound_bags["all"] = 0.05


    #CLEAN CONSTRAINTS -- ONLY INCLUDE CLEAN BAGS
#    pairwise_constraints_new = []
#    lower_diff_bound_bags_new = {}
#    lower_p_bound_bags_new = {}
#    for p in pairwise_constraints:
#        if p[0] in good_bags and p[0] in lower_p_bound_bags:
#            lower_p_bound_bags_new[p[0]] = lower_p_bound_bags[p[0]]
#        if p[1] in good_bags and p[1] in lower_p_bound_bags:
#            lower_p_bound_bags_new[p[1]] = lower_p_bound_bags[p[1]]
#
#        if p[0] in good_bags and p[1] in good_bags:
#            pairwise_constraints_new.append(p)
#            lower_diff_bound_bags_new[p] = lower_diff_bound_bags[p]

    # GET news/not news
    all_news = [v for k,v in bags.items() if "_News" in k]
    all_news = [item for sublist in all_news for item in sublist]
    all_news = list(set(all_news))
    all_ids = set([item for sublist in bags.values() for item in sublist])
    not_news = list(all_ids.difference(all_news))
    X_not_news = X_features[not_news]
    X_news = X_features[all_news]


    if "svm_citation" in model_type:
        y = cites
        from sklearn.linear_model import Ridge
        clf = Ridge()
        clf.fit(X_features, y)
        w_t = clf.coef_.squeeze()
        print(clf.intercept_)
        score = clf.predict(X_features).squeeze()
        info_string ="bp" + use_abstract + model_type

        fpath = 'ballpark_w'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
           pickle.dump(w_t,handle)

        fpath = 'ballpark_score'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(score,handle)

        print(info_string)
        plt = pd.Series(score).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_score"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))

        #SAVES RESULTS FOR HTML
        evaluate(X_features, data, w_t, score, info_string,file_path=file_path,clf=clf, num_to_save=350)

    if "svm_news" in model_type:
        print("svm_news baseline")
        y = [0]*len(X_not_news) + [1]*len(X_news)
        X_features = np.vstack([X_not_news,X_news])

        clf = LinearSVC(random_state=0,class_weight="balanced")
        clf.fit(X_features, y)
        w_t = clf.coef_.squeeze()
        print(clf.intercept_)
        print(classification_report(clf.predict(X_features),y))

        score = clf.decision_function(X_features)

        info_string ="bp" + use_abstract + model_type

        fpath = 'ballpark_w'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
           pickle.dump(w_t,handle)

        fpath = 'ballpark_score'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(score,handle)

        print(info_string)
        plt = pd.Series(score).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_score"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))

        #SAVES RESULTS FOR HTML
        evaluate(X_features, data, w_t, score, info_string,file_path=file_path,clf=clf)

    if "anomaly_detection" in model_type:
        from sklearn.svm import OneClassSVM

        clf = OneClassSVM(kernel='linear')
        clf.fit(X_features)
        w_t = clf.coef_.squeeze()
        print(clf.intercept_)
        score = clf.predict(X_features).squeeze()
        info_string = "bp" + use_abstract + model_type

        fpath = 'ballpark_w' + info_string + '_.pickle'
        with open(str(file_path / fpath), 'wb') as handle:
            pickle.dump(w_t, handle)

        fpath = 'ballpark_score' + info_string + '_.pickle'
        with open(str(file_path / fpath), 'wb') as handle:
            pickle.dump(score, handle)

        print(info_string)
        plt = pd.Series(score).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_score" + info_string + "_HIST.png"
        fig.savefig(str(file_path / fpath))

        # SAVES RESULTS FOR HTML
        evaluate(X_features, data, w_t, score, info_string, file_path=file_path, clf=clf)

    if "dccp" in model_type:
        #ADD DCCP CONTRAINTS -- SUM_TOP_K
        info_string = use_abstract + model_type + use_dccp + use_convex

        max_iter = 10
        loss_dccp_list =[]
        w_t_list = []
        y_t_list = []
        if model_type == "dccp_convex_init":
            CONVEX_INIT = True
            #use original convex ballpark as init for dccp
            #i.e., use weights w and labels y returned from running with dccp=false, use_latent_y=True
            convex_info_string =use_abstract + "latent_y"

            convex_weights_path = 'ballpark_w'+convex_info_string+'_.pickle'
            convex_weights_path = file_path/convex_weights_path
            convex_y_path = 'ballpark_y'+convex_info_string+'_.pickle'
            convex_y_path = file_path/convex_y_path

            ccp_times = 1
            num_trials = 1
        else:
            CONVEX_INIT = False
            convex_weights_path = None
            convex_y_path = None
            ccp_times = 3
            num_trials = 1
        w_t, y_t,loss_dccp = solve_w_y_dccp(X=X_features, pairwise_constraints_indices=pairwise_constraints,
                                        bag_list=bags,upper_p_bound_bags=upper_p_bound_bags,
                                        diff_upper_bound_pairs=upper_diff_bound_bags,
                                        diff_lower_bound_pairs=lower_diff_bound_bags,
                                        top_k_percent=0.1, top_k_bound_ratio=0.9,  # todo: readjust
                                        lower_p_bound_bags=lower_p_bound_bags,
                                        convex_init = CONVEX_INIT,
                                        convex_weights_path = str(convex_weights_path),
                                        convex_y_path = str(convex_y_path),
                                        ccp_times = ccp_times,
                                        max_iter = max_iter)
        score = np.dot(X_features,w_t)

        print(info_string)
        fpath = 'ballpark_w'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
           pickle.dump(w_t,handle)

        fpath = 'ballpark_y'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(y_t,handle)

        fpath = 'ballpark_score'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(score,handle)


        plt = pd.Series(score).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_score"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))
        del plt
        del fig
        plt = pd.Series(y_t).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_y"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))

        evaluate(X_features, data, w_t, y_t, info_string,file_path=file_path)

    #TRAIN MODEL
    elif model_type == "latent_y":
        print("Training model with latent y...")
        w_t,y_t,loss_bp = solve_w_y(X=X_features, pairwise_constraints_indices=pairwise_constraints,
                                    bag_list=bags,upper_p_bound_bags=upper_p_bound_bags,
                                    diff_upper_bound_pairs=upper_diff_bound_bags,
                                    diff_lower_bound_pairs=lower_diff_bound_bags,
                                    lower_p_bound_bags=lower_p_bound_bags)

        #SAVE WEIGHTS, Y TO FILE
        #basic experiment identifier
        info_string =use_abstract + model_type
        score = np.dot(X_features,w_t)

        fpath = 'ballpark_w'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
           pickle.dump(w_t,handle)

        fpath = 'ballpark_y'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(y_t,handle)

        fpath = 'ballpark_score'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(score,handle)

        print(info_string)

        plt = pd.Series(score).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_score"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))
        del plt
        del fig
        plt = pd.Series(y_t).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_y"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))

        #SAVES RESULTS FOR HTML
        evaluate(X_features, data, w_t, y_t, info_string,file_path=file_path)


    elif model_type=="feasibility":
        print("Training feasibility_regression model...")

        w_t = feasibility_regression(X=X_features,
                                     pairwise_constraints_indices=pairwise_constraints,
                                    bag_indices=bags,
                                    upper_p_bound_bags=upper_p_bound_bags,
                                    diff_upper_bound_pairs=upper_diff_bound_bags,
                                    diff_lower_bound_pairs=lower_diff_bound_bags,
                                    lower_p_bound_bags=lower_p_bound_bags)
        #SAVE SCORE TO FILE
        score = np.dot(X_features,w_t)
        #score[good_bags['Not In News']]
        info_string ="bp" + use_abstract + model_type

        fpath = 'ballpark_w'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
           pickle.dump(w_t,handle)

        fpath = 'ballpark_score'+info_string+'_.pickle'
        with open(str(file_path/fpath), 'wb') as handle:
            pickle.dump(score,handle)

        print(info_string)
        plt = pd.Series(score).hist(bins=100)
        fig = plt.get_figure()
        fpath = "ballpark_score"+info_string+"_HIST.png"
        fig.savefig(str(file_path/fpath))

        #SAVES RESULTS FOR HTML
        evaluate(X_features, data, w_t, score, info_string,file_path=file_path)

    ### FOR Y
    print(info_string)
    y_res_path = 'ballpark_results'+info_string+'_y.pickle'
    y_res_path = file_path/y_res_path
    with open(str(y_res_path), 'rb') as handle:
            res = pickle.load(handle)

    html_path = 'ballpark_results'+info_string+'_y_HTML'
    html_path = file_path/html_path
    print(html_path)
    generate_results_html(html_path,res,data)

    #FOR SCORE
    score_res_path = 'ballpark_results'+info_string+'_score.pickle'
    score_res_path = file_path/score_res_path
    with open(str(score_res_path), 'rb') as handle:
            res2 = pickle.load(handle)

    if res==res2:
        print("same res")
    else:
        html_path = 'ballpark_results'+info_string+'_score_HTML'
        html_path = file_path/html_path
        print(html_path)
        generate_results_html(html_path,res2,data)







