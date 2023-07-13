# -*- coding: utf-8 -*-
"""
feasibility_regression: Feasibility method (Problem 7) for Boston dataset, synthetic constraints

solve w_y:  with estimated labels yhat
solve w_y2: see (3), (4) in paper


@author: tomhope
"""

import cvxpy as cp
# import dccp
import numpy as np
import pickle 


# def feasibility_regression(X, pairwise_constraints_indices, 
#                            bag_indices,upper_p_bound_bags,
#                       diff_upper_bound_pairs,diff_lower_bound_pairs,
#                       lower_p_bound_bags):

#     theta = cp.Variable(X.shape[1])
#     reg = cp.square(cp.norm(theta, 2))
    
#     constraints = []
#     added_pairs = []
#     pair_ind = 0
#     for pair in pairwise_constraints_indices:
#         bag_high = bag_indices[pair[0]]
#         bag_low = bag_indices[pair[1]]
      
#         # CHECK FOR EMPTY BAGS
#         if len(bag_high) == 0 or len(bag_low)==0:
#             #print(pair)
#             #print(bag_high)
#             #print(bag_low)
#             break
        
#         scores_high = (1./len(bag_high))*X[bag_high]*theta
#         scores_low = (1./len(bag_low))*X[bag_low]*theta
    
#         if pair in diff_upper_bound_pairs:
#             constraints.append(cp.sum(scores_high) - cp.sum(scores_low) <= diff_upper_bound_pairs[pair])
            
#         if pair in diff_lower_bound_pairs:
#             constraints.append(cp.sum(scores_high) - cp.sum(scores_low) >= diff_lower_bound_pairs[pair])
#         else:
#             constraints.append(cp.sum(scores_high) - cp.sum(scores_low) >= 0)
    
#         if pair[0] not in added_pairs:
#             if pair[0] in upper_p_bound_bags:
#                 constraints.append(cp.sum(scores_high)<=upper_p_bound_bags[pair[0]])
#             if pair[0] in lower_p_bound_bags:
#                 constraints.append(cp.sum(scores_high)>=lower_p_bound_bags[pair[0]])
#             added_pairs.append(pair[0])
#         if pair[1] not in added_pairs:
#             if pair[1] in upper_p_bound_bags:
#                 constraints.append(cp.sum(scores_low)<=upper_p_bound_bags[pair[1]])
#             if pair[1] in lower_p_bound_bags:
#                 constraints.append(cp.sum(scores_low)>=lower_p_bound_bags[pair[1]])
#             added_pairs.append(pair[1])
#         pair_ind+=1

#     # add upper average individual constraints
#     if "all" in upper_p_bound_bags:
#         #print("all")
#         current_bag = bag_indices["all"]
#         #print(len(current_bag))
#         ##print("all upper")
#         #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
#         Xw = X[current_bag]*theta
#         #print(Xw.shape)
#         score_avg = cp.sum((1./len(current_bag))*Xw)
#         #print(score_avg)
#         constraints.append(score_avg <= upper_p_bound_bags["all"])

#     # add upper average individual constraints
#     if "Not In News" in upper_p_bound_bags:
#         #print("Not In News")
#         current_bag = bag_indices["Not In News"]
#         ##print("all upper")
#         #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
#         Xw = X[current_bag]*theta
#         score_avg = cp.sum((1./len(current_bag))*Xw)
#         constraints.append(score_avg <= upper_p_bound_bags["Not In News"])

#     prob = cp.Problem(cp.Minimize(1*reg),constraints = constraints)

#     try:
#         prob.solve(verbose=False)
#     except:
#         prob.solve(solver="SCS")
#     w_t = np.squeeze(np.asarray(np.copy(theta.value)))
#     return w_t        


def solve_w_y(X,pairwise_constraints_indices,bag_list,
                           upper_p_bound_bags={},lower_p_bound_bags={},
                           diff_upper_bound_pairs={},diff_lower_bound_pairs={},
                           overall_upper_bound=1,
                           overall_lower_bound=0,
                           reg_val=10**-1,v = False):
    
    n_test = sum([len(bag) for bag in bag_list.values()])
    print(X.shape[1])
    w = cp.Variable(X.shape[1]) #+intercept
    reg = cp.square(cp.norm(w, 2))
    yhat = cp.Variable(X.shape[0]) #+intercept
    
    constraints = []
    
    constraints.append(yhat>=overall_lower_bound)
    constraints.append(yhat<=overall_upper_bound)

    loss = 0
    added_bags_to_loss = []
    for pair in pairwise_constraints_indices:
        bag_high = bag_list[pair[0]]
        bag_low = bag_list[pair[1]]

        # CHECK FOR EMPTY BAGS
        if len(bag_high) == 0 or len(bag_low)==0:
            continue
        
        if pair[0] not in added_bags_to_loss:
            #print(pair[0])
            
            loss += cp.sum_squares(X[bag_high]*w-yhat[bag_high])
            
            if pair[0] in upper_p_bound_bags: 
                constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) <= upper_p_bound_bags[pair[0]] )
            
            if pair[0] in lower_p_bound_bags: 
                constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) >= lower_p_bound_bags[pair[0]] )

        if pair[1] not in added_bags_to_loss: 
            #print(pair[1])

            loss += cp.sum_squares(X[bag_low]*w-yhat[bag_low])
            
            if pair[1] in upper_p_bound_bags: 
                constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) <= upper_p_bound_bags[pair[1]] )
            
            if pair[1] in lower_p_bound_bags: 
                constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) >= lower_p_bound_bags[pair[1]])
           
        added_bags_to_loss.append(pair[0])
        added_bags_to_loss.append(pair[1])            
    
        if pair in diff_upper_bound_pairs:
            constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) <= diff_upper_bound_pairs[pair])
    
        if pair in diff_lower_bound_pairs:
            constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) >= diff_lower_bound_pairs[pair])
        else:
            constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) >= 0)

    # add upper average individual constraints
    if "all" in upper_p_bound_bags:
        current_bag = bag_list["all"]
        ##print("all upper")
        #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
        score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
        constraints.append(score_avg <= upper_p_bound_bags["all"])

    if "Not In News" in upper_p_bound_bags:
        current_bag = bag_list["Not In News"]
        ##print("all upper")
        #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
        score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
        constraints.append(score_avg <= upper_p_bound_bags["Not In News"])


    if "all"  in lower_p_bound_bags:
        ##print("all lower")
        current_bag = bag_list["all"]
        #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])
        score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
        constraints.append(score_avg >= lower_p_bound_bags["all"])

    prob = cp.Problem(cp.Minimize(loss/n_test + reg_val*reg),constraints = constraints)
    
    try:
       ##print("dccp: ", dccp.is_dccp(prob))
       # #print("dcp: ", prob.is_dcp())
        # prob.solve(method='dccp')
        prob.solve(verbose=v)
       # #print('here')
    except:
        #print('solving with SCS!!!! NOT DCCP!')
        prob.solve(solver="SCS")
    w_t = np.squeeze(np.asarray(np.copy(w.value)))
    y_t = np.squeeze(np.asarray(np.copy(yhat.value)))
    #print("y: ", y_t)
    #print("PROB2 VALUE ------ " ,prob.value)
    return w_t, y_t,prob.value


# def solve_w_y_dccp_convex_start(X,pairwise_constraints_indices,bag_list,
#                            upper_p_bound_bags,lower_p_bound_bags,
#                            diff_upper_bound_pairs,diff_lower_bound_pairs,
#                            top_k_percent, top_k_bound_ratio,
#                            reg_val=10**-1):
#     convex_w = solve_w_y(X, pairwise_constraints_indices, bag_list,upper_p_bound_bags, diff_upper_bound_pairs,
#                          diff_lower_bound_pairs, lower_p_bound_bags)
#     n_test = sum([len(bag) for bag in bag_list.values()])

#     w = cp.Variable(X.shape[1]) #+intercept

#     w.value = np.array(convex_w)

#     reg = cp.square(cp.norm(w, 2))
#     yhat = cp.Variable(X.shape[0]) #+intercept

#     constraints = []
#     added_bags_to_loss = []
#     for pair in pairwise_constraints_indices:

#         bag_high = bag_list[pair[0]]
#         bag_low = bag_list[pair[1]]

#         # CHECK FOR EMPTY BAGS
#         if len(bag_high) == 0 or len(bag_low) == 0:
#             continue

#         if pair[0] not in added_bags_to_loss:

#             if pair[0] in upper_p_bound_bags:
#                 constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) < upper_p_bound_bags[pair[0]] )

#             if pair[0] in lower_p_bound_bags:
#                 constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) > lower_p_bound_bags[pair[0]] )

#         if pair[1] not in added_bags_to_loss:

#             if pair[1] in upper_p_bound_bags:
#                 constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) < upper_p_bound_bags[pair[1]] )

#             if pair[1] in lower_p_bound_bags:
#                 constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) > lower_p_bound_bags[pair[1]])

#         added_bags_to_loss.append(pair[0])
#         added_bags_to_loss.append(pair[1])

#         if pair in diff_upper_bound_pairs:
#             constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) < diff_upper_bound_pairs[pair])

#         if pair in diff_lower_bound_pairs:
#             constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) > diff_lower_bound_pairs[pair])
#         else:
#             constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) > 0)

#     # add upper average individual constraints
#     for bag in upper_p_bound_bags:
#         current_bag = bag_list[bag]
#         # CHECK FOR EMPTY BAGS OR RANDOM BAGS
#         bag_size = int(round(0.05*X.shape[0]))
#         if len(current_bag) == 0 or len(current_bag) == bag_size:
#             continue

#         score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
#         constraints.append(score_avg <= upper_p_bound_bags[bag])

#     for bag in lower_p_bound_bags:
#         current_bag = bag_list[bag]
#         # CHECK FOR EMPTY BAGS
#         if len(current_bag)==0:
#             continue
#         score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
#         constraints.append(score_avg >= lower_p_bound_bags[bag])


#     # add top-k constraints
#     #print('bag list length', len(bag_list))
#     for bag in bag_list:
#         current_bag = bag_list[bag]
#         n_top = int(top_k_percent*len(current_bag))
#         if n_top > 0 and len(current_bag) != 240 and len(current_bag) != 1:  # checks for random bags (size 240) and individual bags
#             #print("percent bag", bag)
#             constraints.append(cp.sum_largest(yhat[current_bag], n_top) >= top_k_bound_ratio*n_top)

#     prob = cp.Problem(cp.Minimize(cp.sum_squares(w - convex_w)), constraints=constraints)

#     ##print("dccp: ", dccp.is_dccp(prob))
#     ##print("dcp: ", prob.is_dcp())
#     prob.solve(method='dccp', ccp_times=1)
#     w_t = np.squeeze(np.asarray(np.copy(w.value)))
#     y_t = np.squeeze(np.asarray(np.copy(yhat.value)))
#     return w_t, y_t,prob.value

# from cvxpy.atoms.sum_largest import sum_largest

# def solve_w_y_dccp(X,pairwise_constraints_indices,bag_list,
#                            upper_p_bound_bags,lower_p_bound_bags,
#                            diff_upper_bound_pairs,diff_lower_bound_pairs,
#                            top_k_percent, top_k_bound_ratio, convex_init=False,
#                            convex_y_path = None,
#                            convex_weights_path = None,
#                            reg_val=10**-1,ccp_times = 1, max_iter = 10, use_mech = True):

#     n_test = sum([len(bag) for bag in bag_list.values()])

#     w = cp.Variable(X.shape[1]) #+intercept
#     reg = cp.square(cp.norm(w, 2))
#     yhat = cp.Variable(X.shape[0]) #+intercept

#     # convex intialise
#     if convex_init:
#         '''
#         convex_w, convex_y = solve_w_y(X, pairwise_constraints_indices, bag_list,upper_p_bound_bags, diff_upper_bound_pairs,
#                              diff_lower_bound_pairs, lower_p_bound_bags)
#         '''
#         if use_mech:
#             mech_string = ""
#         else:
#             mech_string = "NO_MECH"
#         with open(convex_y_path, 'rb') as handle:
#             convex_y = pickle.load(handle)

#         with open(convex_weights_path, 'rb') as handle:
#             convex_w = pickle.load(handle)#,encoding='iso-8859-1')
#             #print(convex_w)
#             #print(convex_w.shape)
#         w.value = np.array(convex_w)
#         yhat.value = np.array(convex_y)

#     constraints = []
    
#     constraints.append(yhat>=0)
#     constraints.append(yhat<=1)
    
#     loss = 0
#     added_bags_to_loss = []
#     for pair in pairwise_constraints_indices:

#         bag_high = bag_list[pair[0]]
#         bag_low = bag_list[pair[1]]

#         # CHECK FOR EMPTY BAGS
#         if len(bag_high) == 0 or len(bag_low)==0:
#             continue

#         if pair[0] not in added_bags_to_loss:
#             loss += cp.sum_squares(X[bag_high]*w-yhat[bag_high])

#             if pair[0] in upper_p_bound_bags:
#                 constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) <= upper_p_bound_bags[pair[0]] )

#             if pair[0] in lower_p_bound_bags:
#                 constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) >= lower_p_bound_bags[pair[0]] )


#         if pair[1] not in added_bags_to_loss:
#             loss += cp.sum_squares(X[bag_low]*w-yhat[bag_low])

#             if pair[1] in upper_p_bound_bags:
#                 constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) <= upper_p_bound_bags[pair[1]] )

#             if pair[1] in lower_p_bound_bags:
#                 constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) >= lower_p_bound_bags[pair[1]])

#         added_bags_to_loss.append(pair[0])
#         added_bags_to_loss.append(pair[1])

#         if pair in diff_upper_bound_pairs:
#             constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) <= diff_upper_bound_pairs[pair])

#         if pair in diff_lower_bound_pairs:
#             constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) >= diff_lower_bound_pairs[pair])
#         else:
#             constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) >= 0)


#     if "all" in upper_p_bound_bags:
#         current_bag = bag_list["all"]
#         ##print("all upper")
#         #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
#         score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
#         constraints.append(score_avg <= upper_p_bound_bags["all"])

#     if "Not In News" in upper_p_bound_bags:
#         current_bag = bag_list["Not In News"]
#         ##print("all upper")
#         #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
#         score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
#         constraints.append(score_avg <= upper_p_bound_bags["Not In News"])



#     if "all"  in lower_p_bound_bags:
#         ##print("all lower")
#         current_bag = bag_list["all"]
#         #loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])
#         score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
#         constraints.append(score_avg >= lower_p_bound_bags["all"])

# #    # add upper average individual constraints
# #    for bag in upper_p_bound_bags:
# #        current_bag = bag_list[bag]
# #        # CHECK FOR EMPTY BAGS OR RANDOM BAGS
# #        bag_size = int(round(0.05*X.shape[0]))
# #        if len(current_bag) == 0 or len(current_bag) == bag_size:
# #            continue
# #
# #        loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])  # todo add only if not added
# #        score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
# #        constraints.append(score_avg <= upper_p_bound_bags[bag])
# #
# #    for bag in lower_p_bound_bags:
# #        current_bag = bag_list[bag]
# #        # CHECK FOR EMPTY BAGS
# #        if len(current_bag)==0:
# #            continue
# #        loss += cp.sum_squares(X[current_bag]*w-yhat[current_bag])
# #        score_avg = (1./len(current_bag))*cp.sum(yhat[current_bag])
# #        constraints.append(score_avg >= lower_p_bound_bags[bag])

#     # add top-k constraints
#     #print('bag list length', len(bag_list))
#     for bag in bag_list:
#         current_bag = bag_list[bag]
#         n_top = int(top_k_percent*len(current_bag))
#         if n_top > 0 and len(current_bag)> 1:  # checks for individual bags
#             #print("percent bag", bag)
#             constraints.append(sum_largest(yhat[current_bag], n_top) >= top_k_bound_ratio*n_top)

#     prob = cp.Problem(cp.Minimize(loss/n_test + reg_val*reg), constraints=constraints)

#     ##print("dccp: ", dccp.is_dccp(prob))
#     ##print("dcp: ", prob.is_dcp())
#     result  = prob.solve(method='dccp', ccp_times=ccp_times, max_iter=max_iter,verbose=False)
#     ##print('here_dccp')
#     #print("PROB2 VALUE ------ " ,result[0])
#     ##print(prob.status)


#     '''
#     try:
#         #print("dccp: ", dccp.is_dccp(prob))
#         #print("dcp: ", prob.is_dcp())
#         prob.solve(method='dccp', ccp_times=1, tau=0.15)
#         #print('here')
#         #prob.solve(verbose = v)
#     except:
#         #print('solving with SCS!!!! NOT DCCP!')
#         prob.solve(solver="SCS")
#     '''
#     w_t = np.squeeze(np.asarray(np.copy(w.value)))
#     y_t = np.squeeze(np.asarray(np.copy(yhat.value)))
#     return w_t, y_t, result[0]


def solve_w_y2(X,pairwise_constraints_indices,bag_list,
                           upper_p_bound_bags,lower_p_bound_bags,
                           diff_upper_bound_pairs,diff_lower_bound_pairs,
                           reg_val=10**-1,v = False):
    
    n_test = sum([len(bag) for bag in bag_list.values()])

    yhat = cp.Variable(X.shape[0]) #+intercept
    
    #see (3), (4) in paper
    w = np.linalg.pinv(reg_val*np.eye(N=X.shape[1]) + (X.T).dot(X))
    w = w.dot(X.T)*yhat
    

    constraints = []
    loss = 0
    pair_ind = 0
    added_bags_to_loss = []
    for pair in pairwise_constraints_indices:
        
        bag_high = bag_list[pair[0]]
        bag_low = bag_list[pair[1]]
    

        if pair[0] not in added_bags_to_loss:
            loss += cp.sum_squares(X[bag_high]*w-yhat[bag_high])
            
            if pair[0] in upper_p_bound_bags: 
                constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) < upper_p_bound_bags[pair[0]] )
            
            if pair[0] in lower_p_bound_bags: 
                constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) > lower_p_bound_bags[pair[0]] )
            
            
        if pair[1] not in added_bags_to_loss: 
            loss += cp.sum_squares(X[bag_low]*w-yhat[bag_low])
            
            if pair[1] in upper_p_bound_bags: 
                constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) < upper_p_bound_bags[pair[1]] )
            
            if pair[1] in lower_p_bound_bags: 
                constraints.append((1./(len(bag_low)))*cp.sum(yhat[bag_low]) > lower_p_bound_bags[pair[1]])
           
        added_bags_to_loss.append(pair[0])
        added_bags_to_loss.append(pair[1])            
    
        if pair in diff_upper_bound_pairs:
            constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) < diff_upper_bound_pairs[pair])
    
        if pair in diff_lower_bound_pairs:
            constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) > diff_lower_bound_pairs[pair])
        else:
            constraints.append((1./(len(bag_high)))*cp.sum(yhat[bag_high]) - (1./(len(bag_low)))*cp.sum(yhat[bag_low]) > 0)
    
        pair_ind+=1    
    
    prob = cp.Problem(cp.Minimize(loss/n_test ),constraints = constraints)
    
    try:
        prob.solve()
    except:
        prob.solve(solver="SCS")
    w_t = np.squeeze(np.asarray(np.copy(w.value)))
    #print("PROB VALUE ------ " ,prob.value)
    return(w_t)

