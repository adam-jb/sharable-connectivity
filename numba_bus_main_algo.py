

# # Returns more outputs than standard bus algo function, Good for exploration and plotting

# # Shows that lots of transitions are being taken (up to 10, even in a 1 hr time limit!): how to penalise for these?
# # And 20 transitions in 2 hours. Unless they're being counted wrong



import os
import json
import copy
import numpy as np
import random
import string
from numba import jit, njit, objmode
from numba.typed import Dict
from numba.core.types import int32, int64, int8
import pickle5 # to read
import sys
import time
import pickle # to export
import pandas as pd





@njit  # inc njit so func can be called by other funcs in this script
def diminishing_returns(purpose, total_value_visited):
    
    """
    Function to calculate the diminishing returns of locations visited, for each purpose
    
    Returns a multiplier to apply to location (1 is the highest possible)
    
    Diminishing returns should be proportional to the total value of that thing visited, rather than number of places:
    because some places might meet some needs (eg: supermarket), reducing the need for visiting other places
    
    purpose: purpose of location visited
    total_value_visited: total value of things of that type visited
    
    eg: if 10th Shop visited so far -> diminishing_returns('Shopping', 10)
    ## Except that 'Shopping' would be represented by an integer
    
    
    1 = Business
    2 = Education
    3 = Entertain / public activity
    4 = Shopping
    5 = Visit friends at private home
    
    
    THE PARAMETERS HERE NEED CALIBRATING
    
    """
    
    if purpose == 1:
        return 1
    if purpose == 2:
        return 1**total_value_visited # previously a float instead of 1. Would need to cast total_value_visited to a float
    if purpose == 3:
        return 1
    if purpose == 4:
        return 1**total_value_visited  # previoulsy a float instead of 1. Would need to cast total_value_visited to a float
    else:
        return 1   # for visiting friends (purpose 5)

    

@njit
def get_pos_in_listheap(elements, new_value):
    """
    elements = list of numbers to find the position in
    
    new_value
    """
    
    #print(f'type(new_value): {type(new_value)}')
    
    elem_count = len(elements)
    bisections_to_do = int(np.ceil(np.log2(elem_count)))
    position_tracker = int(0)
    

    for i in range(bisections_to_do):        
        
        cut_pos = 2**(bisections_to_do-1-i)
        
        idx = position_tracker + cut_pos
        
        if elem_count <= idx: # only do this if idx fits in existing list
            cut_pos = 0
        
        try:
            if new_value >= elements[position_tracker + cut_pos]:
                position_tracker += cut_pos    # starting point moves to middle of that section
            else:
                pass
        except:
            pass
        
        
    ## correction maker: makes it work. Less elegant but hey
    for i in range(1):
        if elements[position_tracker] < new_value:
            position_tracker += 1

            
    return position_tracker





@njit
def get_value_all_purposes(location_value_array, time_travelled, time_travelled_value_array, nth_of_purpose_array):
        
    """
     Docs for  get_value_all_purposes()
        
    Gets the value for a node according to all purposes
    
    Uses diminishing returns

    
    
    location_value_array = int32 list. Standalone values of that node: list of 5, covering each purpose
    
    time_travelled = int32. Time to get to the node (or generalised cost)        

    time_travelled_value_array = array for time taken to travel
    the array is from distribution of distances people take with that purpose according to NTS

    nth_of_purpose_array = array of 5 numbers, each of which iteration of thing is reached. If many things are reached the midpoint is taken
    
    """
    
    ## LOOP THROUGH AND GET VALUE FOR EACH PURPOSE
    for i in range(5):
        location_value = location_value_array[i]
        
        if location_value == 0:  # calling dict below would get key error in this case
            pass
            
        else:
            
            nth_of_purpose_array[i] += location_value * (1 + 19 - get_pos_in_listheap(time_travelled_value_array[i,:], time_travelled)) * diminishing_returns(i, nth_of_purpose_array[i])
            
    
    return nth_of_purpose_array




#### Set of functions which cast results from numpy int types to python int types. No longer used
@njit
def get_arrival_time_next_stop(a, b, c):
    return a + b + c


@njit
def get_time_travelled(a, b):
    return a + b


@njit
def update_time_so_far(time_so_far):
    return time_so_far


@njit
def insert_to_ticker(ticker, ix, val_value):
    ticker.insert(ix, val_value) # val_array[i, 1])
    return 0



## func slots into main_bus_algo. Does everything inplace. Does everything related to taking a bus route.
@njit
def bus_function(p2_nodes_typed_dict,
                 p1_nodes_typed_dict,
                 val,
                 travel_time_ticker, 
                 time_so_far,
                 val_array, 
                 ticker,
                 route_arrived_by_ticker,
                 maximum_travel_time,
                 TripStartSeconds,
                 route_arrived_by,
                 zero_int32,
                 one_int32,
                 start_node_tracker,
                 end_node_tracker,
                 travel_times_all_for_plot,
                 link_type_tracker,
                 transition_ticker,
                 transition_count,
                 transition_tracker,
                 nodes_visited_ticker,
                 times_travelled_ticker,
                 modes_taken_ticker,
                 nodes_visited_store,
                 times_travelled_store,
                 modes_taken_store,
                 nodes_visited,
                 times_travelled,
                 modes_taken,
                 wait_time_store,
                 next_leaving_time_store,
                 time_of_arrival_current_node_store,
                 route_journey_time_store,
                 time_so_far_store,
                 arrives_exactly_as_service_leaves,
                 p1_record_set_nodes_visited,
                 p2_record_set_nodes_visited):

    
    
    """
    val_array = walking transition matrix. Where first row isnt part of main matrix
    
    time_so_far = time travelled up to this point (from start of simulation time, not from midnight)
    
    val = node id being visited
    
    zero_int32 and one_int32 are here because numba cant infer the right types (at least it cant when the main function isnt jitted)
    
    
    """
    
        
    if val in p2_record_set_nodes_visited:
        raise ValueError(f'p2_nodes_typed_dict[val] didnt find a key for val {val}: if there is no key in the dictionary (p1 or p2) it shouldve been caught in the main algo function')
        """
        ticker.pop(0)
        travel_time_ticker.pop(0)
        route_arrived_by_ticker.pop(0)
        transition_ticker.pop(0)
        nodes_visited_ticker.pop(0)
        times_travelled_ticker.pop(0)
        modes_taken_ticker.pop(0)
        """
    else:
        p2_record_set_nodes_visited.add(val)
        bus_array = p2_nodes_typed_dict[val]


    
    next_leaving_times = bus_array[1:,0]  # 1st col has leaving times, 2nd has travel times



    # find time of arrival in seconds past midnight
    time_of_arrival_current_node = TripStartSeconds + time_so_far  # time we get to current node
    
        


    # track whether a service can be found after the current time
    found_next_service = 0      # ensure default value is zero
    for service_ix in range(len(next_leaving_times)):
        if time_of_arrival_current_node <= next_leaving_times[service_ix]:
            next_leaving_time = next_leaving_times[service_ix]
            found_next_service = 1
            service_ix_found = service_ix
            arrives_exactly_as_service_leaves.append(int(time_of_arrival_current_node==next_leaving_times[service_ix]))
            break   # terminate loop having found first service which comes after the current time. Its departure time is stored as next_leaving_time

            
            
       
    # only do the rest if another service was found
    if found_next_service == 1:

        wait_time = next_leaving_time - time_of_arrival_current_node

        # interchange penalty ** CURRENTLY **NOT** APPLYING THIS TO THE 75mins limit from 95th percentile in NTS
        #if route_arrived_by == 0:  # if route_arrived_by == 1, this route's bus was taken to this stop, so no interchange, 
                                    # otherwise we apply the interchange penalty mulitplier

            #wait_time = wait_time * 2  # UNCOMMENT THIS TO FACTOR THE MULTIPLIER IN TO THE 75MINS LIMT



        # arrival_time_next_stop = seconds since start of simulation
        #arrival_time_next_stop = get_arrival_time_next_stop(time_so_far, wait_time, bus_array[1:,1][service_ix_found]) # bus_array[1:,1] = travel time of next journey
        
        arrival_time_next_stop = time_so_far + wait_time + bus_array[1:,1][service_ix_found]  # bus_array[1:,1] has travel times

        # only add stop to ticker if its in an acceptable time window
        if arrival_time_next_stop < maximum_travel_time:


            ## This try/catch wrapper might be unnecessary, as 'found_next_service' might cause us to skip nodes with no following stops
            try:
                #next_node = bus_array[0,0]
                            
                # store results
                ix = get_pos_in_listheap(travel_time_ticker, arrival_time_next_stop) 
                ticker.insert(ix, bus_array[0,0])  
                #insert_to_ticker(ticker, ix, bus_array[0,0])
                travel_time_ticker.insert(ix, arrival_time_next_stop)  # THIS IS THE LINE which causes the error in numba when casting isnt explicit
                #first_vals_in_pair.insert(ix, val)
                #route_arrived_by_ticker.insert(ix, one_int32)  # APRIL CHANGE
                route_arrived_by_ticker.insert(ix, 1)
                
                start_node_tracker.append(val)
                end_node_tracker.append(bus_array[0,0])
                travel_times_all_for_plot.append(arrival_time_next_stop)
                link_type_tracker.append(1)
                
                
                nodes_visited_ticker.insert(ix, nodes_visited + [val])
                times_travelled_ticker.insert(ix, times_travelled + [arrival_time_next_stop])
                modes_taken_ticker.insert(ix, modes_taken + [1])      

                nodes_visited_store.append(nodes_visited + [val])
                times_travelled_store.append(times_travelled + [arrival_time_next_stop])
                modes_taken_store.append(modes_taken + [1])
                
                
                
                wait_time_store.append(wait_time)
                next_leaving_time_store.append(next_leaving_time)
                time_of_arrival_current_node_store.append(time_of_arrival_current_node)
                route_journey_time_store.append(bus_array[1:,1][service_ix_found])
                time_so_far_store.append(time_so_far)
                
                

                
                # route_arrived_by == 0 means someone was getting on a service; route_arrived_by == 1 means they're already on the service
                if route_arrived_by == 0:
                    transition_ticker.insert(ix, transition_count + 1)
                    transition_tracker.append(transition_count + 1)
                else:
                    transition_ticker.insert(ix, transition_count)
                    transition_tracker.append(transition_count)
                    
                
                #if route_arrived_by == 3:
                #    services_taken_count_ticker.insert(ix, services_taken_count + 1)
                #else:
                #    services_taken_count_ticker.insert(ix, services_taken_count)
                


            except:
                pass
                #print('next node couldnt be found'


    """
    # delete bus node (doing this within the function because only delete for p2 visits
    p2_dict_to_recreate[val] = p2_nodes_typed_dict[val]
    del p2_nodes_typed_dict[val]
    """
    
  
    return 0



# declare this here to be used to initialise dict_to_recreate. Need to be declared outside a jitted function
int32_array = int32[:,:]
int32_1d_array = int32[:]



@njit
def return_int_of_val(x):
    """ Casts int32 to int, no longer needed """
    return x




@njit  # remove numba to print. Printing doesn't work with numba
def main_bus_algo(p1_nodes_typed_dict, 
              p2_nodes_typed_dict,
              node_values,    # previously values_vector
              start_node_ix,  
              travel_time_relationships,
              init_travel_time,
              TripStartSeconds,
              max_travel_time):
    
    """Traversal simulation
    
    Don't record start_end_node_pairs to cut nearly half of the runtime (doing this also eliminates
    need to record previous_value and first_vals_in_pair)
    
    max_travel_time: set to 0 to let people go as far as NTS allows them to; otherwise sets maximum seconds they can travel
    
    """
    

    
    
    maximum_travel_time = np.max(travel_time_relationships) + 1
    if max_travel_time > 10:
        maximum_travel_time = max_travel_time
    
    
    ## to append to dict after previous run. Only the first dict has things dropped from it
    """
    p1_dict_to_recreate = Dict.empty(
        key_type=int32,
        value_type=int32_array,  # specify int32_array outside the function
    ) 
    
    p2_dict_to_recreate = Dict.empty(
        key_type=int32,
        value_type=int32_array,  # specify int32_array outside the function
    ) 
    """
    
    
    p1_record_set_nodes_visited = {1}
    p1_record_set_nodes_visited.pop()
    
    p2_record_set_nodes_visited = {1}
    p2_record_set_nodes_visited.pop()
    
    

    all_purposes_values = [0, 0, 0, 0, 0]

    
        
    ticker = [int32(start_node_ix)]    # which to visit next. Error without int32
    travel_time_ticker = [init_travel_time]   # how far travelled so far
    first_vals_in_pair = [start_node_ix]  # node travelled from
    route_arrived_by_ticker = [0]        # 0 for arrived by walk; 1 for arrived by public transport
    transition_ticker = [0]
    
    
    #services_taken_count_ticker = [0]    # number of services taken in total. Don't allow more than X to be used
    
    

    #total_value = 0  # storing value for each purpose separately now, as it is used in diminishing returns calc
    iters = 0

    ### Comment this out if not using them later
    start_node_tracker = [0]
    end_node_tracker = [0]
    travel_times_all_for_plot = [0]    # time arrived at node
    link_type_tracker = [0] 
    transition_tracker = [0]
    
    
    # could have exception handling for type of object being the input; raising specific errors for each section
    # 
    types_iter = [0] 
    
    
    # storing places visited, times to get there, modes taken
    nodes_visited_store = [[0]]
    times_travelled_store = [[0]]
    modes_taken_store = [[0]]
    
    
    # storing things as tickers to be used
    nodes_visited_ticker = [[0]]
    times_travelled_ticker = [[0]]
    modes_taken_ticker = [[0]]   # this exists purely to populate the modes_taken_store
    
    
    # transitions
    transition_count_store = [0]
    
    
    
    wait_time_store = [0]
    next_leaving_time_store = [0]
    time_of_arrival_current_node_store = [0]
    route_journey_time_store = [0]
    time_so_far_store = [0]
    
    
    arrives_exactly_as_service_leaves = [0]
    
    
    
    ## main loop
    while True:
        
        #start_of_iter = time.time()
        
        # extract first values in tickers. Simulation ends when ticker has run out
        try:
            val = ticker[0]
            time_so_far = travel_time_ticker[0]
            #previous_value = first_vals_in_pair[0]
            route_arrived_by = route_arrived_by_ticker[0]
            transition_count = transition_ticker[0]
            #services_taken_count = services_taken_count_ticker[0]
            
            nodes_visited = nodes_visited_ticker[0]
            times_travelled = times_travelled_ticker[0]
            modes_taken = modes_taken_ticker[0]
            
        except:
            break
            

            
            
         # catching nodes already visited
        if val in p1_record_set_nodes_visited:
            ticker.pop(0)
            travel_time_ticker.pop(0)
            route_arrived_by_ticker.pop(0)
            transition_ticker.pop(0)
            nodes_visited_ticker.pop(0)
            times_travelled_ticker.pop(0)
            modes_taken_ticker.pop(0)
            continue
        else:
            p1_record_set_nodes_visited.add(val)
            
        
        
        # extract tuple of values, or skip if it's not in the dict (ie, if it's already been visited)
        try:
            val_array = p1_nodes_typed_dict[val] 
        except:
            ticker.pop(0)
            travel_time_ticker.pop(0)    
            route_arrived_by_ticker.pop(0)   
            transition_ticker.pop(0)
            nodes_visited_ticker.pop(0)
            times_travelled_ticker.pop(0)
            modes_taken_ticker.pop(0) 
            
            continue
            
            
            
        ##### TO CHECK IN NON-BUS ALGO! USE id_new_node instead of val in the code 2 lines below  
        # store value of the node. Only run for walking nodes
        if val < 40_000_000:
            all_purposes_values = get_value_all_purposes(node_values[val,:],  # takes all_purposes_values and updates it
                                      time_so_far, 
                                      travel_time_relationships, 
                                      all_purposes_values)
        
        
        
 

        # record walking if node is a bus-walking-subnode or walking node
        if val_array[0, 0] == 0:
            
            types_iter.append(0)
            
            for i in range(1, len(val_array)):   # first row has the classification of node; start iterating from 2nd node
                
                time_travelled = time_so_far + val_array[i, 0]

                if time_travelled < maximum_travel_time:                  

                    ix = get_pos_in_listheap(travel_time_ticker, time_travelled) # 31st march: issue here, likely due to travel_time_ticker insertion in bus algo
                    ticker.insert(ix, val_array[i, 1])
                    travel_time_ticker.insert(ix, time_travelled)
                    route_arrived_by_ticker.insert(ix, 0)  # REMOVE INT32 CASTING WHEN MAIN ALGO IS JITTED 
                    transition_ticker.insert(ix, transition_count)
                    #services_taken_count_ticker.insert(ix, services_taken_count)  # preserves number of services taken
                    
                    nodes_visited_ticker.insert(ix, nodes_visited + [val])
                    times_travelled_ticker.insert(ix, times_travelled + [time_travelled])
                    modes_taken_ticker.insert(ix, modes_taken + [0])
                    
                    start_node_tracker.append(val)
                    end_node_tracker.append(val_array[i, 1])
                    travel_times_all_for_plot.append(time_travelled)
                    link_type_tracker.append(0)
                    transition_tracker.append(transition_count)
                    
                    nodes_visited_store.append(nodes_visited + [val])
                    times_travelled_store.append(times_travelled + [time_travelled])
                    modes_taken_store.append(modes_taken + [0])
                    
                    
                    ## these are for buses only. Adding data to keep in sync with other indexes
                    wait_time_store.append(0)
                    next_leaving_time_store.append(0)
                    time_of_arrival_current_node_store.append(0)
                    route_journey_time_store.append(0)
                    time_so_far_store.append(0)
                    
                    arrives_exactly_as_service_leaves.append(0)

                    
            """
            p1_dict_to_recreate[val] = p1_nodes_typed_dict[val]   # store the dict to recreate
            del p1_nodes_typed_dict[val]                          # delete node from dict
            """
    
    
                    
        # record if node is a walking-bus-subnode
        if val_array[0, 0] == 1 or val_array[0, 0] == 2:
            
            types_iter.append(2)
            
            if time_so_far < maximum_travel_time:
                            
                # record walking to other subnodes for that stop
                for i in range(1, len(val_array)):
                
                    ##### Note that none of the other subnotes for the stop will have value as destinations, so skip calling get_value_all_purposes()
                
                    ix = get_pos_in_listheap(travel_time_ticker, time_so_far)  #time_travelled)  
                    ticker.insert(ix, val_array[i, 1])
                    #insert_to_ticker(ticker, ix, val_array[i, 1])
                    travel_time_ticker.insert(ix, time_so_far)
                    #first_vals_in_pair.insert(ix, val)
                    route_arrived_by_ticker.insert(ix, 0)    # REMOVE INT32 CASTING WHEN MAIN ALGO IS JITTED
                    transition_ticker.insert(ix, transition_count)
                    #services_taken_count_ticker.insert(ix, services_taken_count)
                    
                    nodes_visited_ticker.insert(ix, nodes_visited + [val])
                    times_travelled_ticker.insert(ix, times_travelled + [time_travelled])
                    modes_taken_ticker.insert(ix, modes_taken + [2])
                    
                    start_node_tracker.append(val)
                    end_node_tracker.append(val_array[i, 1])
                    travel_times_all_for_plot.append(time_so_far)
                    link_type_tracker.append(2)
                    transition_tracker.append(transition_count)
                    
                    nodes_visited_store.append(nodes_visited + [val])
                    times_travelled_store.append(times_travelled + [time_travelled])
                    modes_taken_store.append(modes_taken + [0])
                    
                    
                    
                    ## these are for buses only. Adding data to keep in sync with other indexes
                    wait_time_store.append(0)
                    next_leaving_time_store.append(0)
                    time_of_arrival_current_node_store.append(0)
                    route_journey_time_store.append(0)
                    time_so_far_store.append(0)
                    
                    arrives_exactly_as_service_leaves.append(0)
                        
                    
            """
            p1_dict_to_recreate[val] = p1_nodes_typed_dict[val]   # store the dict to recreate
            del p1_nodes_typed_dict[val]                          # delete node from dict
            """
             

                
                
        # record if node is a route-bus-subnode (denoted by value of 1 in val_array[0,0])
        # these nodes go only to the metanode or another bus node on the same route
        # services_taken_count is added to every time a service is taken from a metanode
        #if val_array[0, 0] == 1  and services_taken_count <= 3:
        if val_array[0, 0] == 1:
            
    
            types_iter.append(1)


            bus_function(p2_nodes_typed_dict,
                 p1_nodes_typed_dict,
                 val,
                 travel_time_ticker, 
                 time_so_far,
                 val_array, 
                 ticker,
                 route_arrived_by_ticker,
                 maximum_travel_time,
                 TripStartSeconds,
                 route_arrived_by,
                 0,
                 1,
                 start_node_tracker, 
                 end_node_tracker,
                 travel_times_all_for_plot,
                 link_type_tracker,
                 transition_ticker,
                 transition_count, 
                 transition_tracker,
                 nodes_visited_ticker,
                 times_travelled_ticker,
                 modes_taken_ticker,
                 nodes_visited_store,
                 times_travelled_store,
                 modes_taken_store,
                 nodes_visited,
                 times_travelled,
                 modes_taken,
                 wait_time_store,
                 next_leaving_time_store,
                 time_of_arrival_current_node_store,
                 route_journey_time_store,
                 time_so_far_store,
                 arrives_exactly_as_service_leaves,
                 p1_record_set_nodes_visited,
                 p2_record_set_nodes_visited)

            
                 #services_taken_count_ticker,
                 #services_taken_count)
            
    
        # drop old things and store previous value
        ticker.pop(0)
        travel_time_ticker.pop(0)
        route_arrived_by_ticker.pop(0)
        transition_ticker.pop(0)
        #services_taken_count_ticker.pop(0)
        previous_val = val
        
        
        nodes_visited_ticker.pop(0)
        times_travelled_ticker.pop(0)
        modes_taken_ticker.pop(0)

        
        iters += 1
        
        
    ## Simulation has now ended
    """
    
    # recreate dicts. Much faster to do this inside a jitted func
    for k, v in p1_dict_to_recreate.items():
        p1_nodes_typed_dict[k] = v
    
    for k, v in p2_dict_to_recreate.items():
        p2_nodes_typed_dict[k] = v
    """
        
        
    # route_arrived_by is only used to determine if arrived by metanode and thus should +1 services_taken_count
    
    # services_taken_count is services_taken_count of most recent iter only
    return all_purposes_values, iters , travel_times_all_for_plot, start_node_tracker, end_node_tracker, link_type_tracker, transition_tracker, nodes_visited_store, times_travelled_store, modes_taken_store, wait_time_store, next_leaving_time_store, time_of_arrival_current_node_store, route_journey_time_store, time_so_far_store, arrives_exactly_as_service_leaves




