# -*- coding: latin-1 -*-
import numpy as np

################################################
### CALCULAR EL VALOR DE LA FUNCION OBJETIVO ###
################################################

# calcula fitness del cromosoma de 0s y 1s
def objective_value(chromosome,iteration_i):  
    
    lb_x = -6 # limite inferior del cromosoma x
    ub_x = 6 # limite superior del cromosoma x
    len_x = (len(chromosome)//2) # long. cromosoma x
    lb_y = -6 # limite inferior del cromosoma y
    ub_y = 6 # limite superior del cromosoma y
    len_y = (len(chromosome)//2) # long. cromosoma y
    
    precision_x = (ub_x-lb_x)/((2**len_x)-1) # precision para x
    precision_y = (ub_y-lb_y)/((2**len_y)-1) # precision para y
    
    z = 0 # empezamos en 2^0
    t = 1 # empezamos al final del vector [index -1]
    x_bit_sum = 0 # initiacion (sum(bit)*2^i es 0 al inicio)
    for i in range(len(chromosome)//2):
        x_bit = chromosome[-t]*(2**z)
        x_bit_sum = x_bit_sum + x_bit
        t = t+1
        z = z+1   
    
    z = 0 
    t = 1 + (len(chromosome)//2) # [6,8,3,9] (primeros 2 son y, asi que index será 1+2 = -3)
    y_bit_sum = 0 # initiacion (sum(bit)*2^i es 0 al inicio)
    for j in range(len(chromosome)//2):
        y_bit = chromosome[-t]*(2**z)
        y_bit_sum = y_bit_sum + y_bit
        t = t+1
        z = z+1
    
    # formulas para decodificar 0s y 1s a un numero real, x o y
    decoded_x = (x_bit_sum*precision_x)+lb_x
    decoded_y = (y_bit_sum*precision_y)+lb_y
    
    # func. himmelblau 
    # min ((x**2)+y-11)**2+(x+(y**2)-7)**2
    # func objetivo
    
    
    g_restriction_value = 9 - decoded_x - decoded_y
    
    if g_restriction_value > 0:
        
        penalization = 0
        
        obj_function_simple=((decoded_x**2)+decoded_y-11)**2+(decoded_x+(decoded_y**2)-7)**2
        
        obj_function_value = obj_function_simple + penalization
        
    else:
        
        penalization = (0.5*iteration_i)*(9 - decoded_x - decoded_y)**2
        
        obj_function_simple = ((decoded_x**2)+decoded_y-11)**2+(decoded_x+(decoded_y**2)-7)**2
        
        obj_function_value = obj_function_simple + penalization
    
    return decoded_x,decoded_y,obj_function_value, obj_function_simple, penalization# x,y, func obj


#################################################
### ESCOGER 2 PADRES                          ###
### USANDO TORNEO                     ###########
#################################################

def find_parents_ts(all_solutions,iteration_i):
    
    # array vacio para los padres
    parents = np.empty((0,np.size(all_solutions,1)))
    
    for i in range(2): # son 2 padres
        
        # escoger 3 posibles padres aleatorios        
        # con 3 enteros random
        indices_list = np.random.choice(len(all_solutions),3,replace=False)
        
        # obtener los padres posibles
        posb_parent_1 = all_solutions[indices_list[0]]
        posb_parent_2 = all_solutions[indices_list[1]]
        posb_parent_3 = all_solutions[indices_list[2]]
        
        # obtener su func obj.
        obj_func_parent_1 = objective_value(posb_parent_1,iteration_i)[2] # possible parent 1
        obj_func_parent_2 = objective_value(posb_parent_2,iteration_i)[2] # possible parent 2
        obj_func_parent_3 = objective_value(posb_parent_3,iteration_i)[2] # possible parent 3
        
        # hallar el mejor
        min_obj_func = min(obj_func_parent_1,obj_func_parent_2,obj_func_parent_3)
        
        if min_obj_func == obj_func_parent_1:
            selected_parent = posb_parent_1
        elif min_obj_func == obj_func_parent_2:
            selected_parent = posb_parent_2
        else:
            selected_parent = posb_parent_3
        
        # poner el mejor en el array
        parents = np.vstack((parents,selected_parent))
        
    parent_1 = parents[0,:] # parent_1, 1er elemento del array
    parent_2 = parents[1,:] # parent_2, 2do elemento del array
    
    return parent_1,parent_2 # devuelve 2 arrays, cada array es un padre



####################################################
### CROSSOVER 2 padres --> 2 hijos               ###
####################################################

# inputs: parent_1, parent_2,  y la prob. de crossover
# probabilidad default es 1
def crossover(parent_1,parent_2,prob_crsvr=1):
    
    child_1 = np.empty((0,len(parent_1)))
    child_2 = np.empty((0,len(parent_2)))
    
    
    rand_num_to_crsvr_or_not = np.random.rand() # se hace o no se hace crossover???
    
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(0,len(parent_2))
        
        # diferentes indices
        # para que haya crossover de por lo menos 1 gen
        while index_1 == index_2:
            index_2 = np.random.randint(0,len(parent_1))
        
        
        # SI el index de PARENT_1 es antes del de PARENT_2
        # ej: parent_1 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        # ej: parent_2 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        if index_1 < index_2:
            
            # PADRE 1          
            first_seg_parent_1 = parent_1[:index_1]
            
            mid_seg_parent_1 = parent_1[index_1:index_2+1]
            
            last_seg_parent_1 = parent_1[index_2+1:]
            
            
            # PADRE 2
            
            first_seg_parent_2 = parent_2[:index_1]
            
            mid_seg_parent_2 = parent_2[index_1:index_2+1]
            
            last_seg_parent_2 = parent_2[index_2+1:]
            
            
            ### HIJO 1 ###

            # 1er segmento de parent_1
            # 2do segmento de parent_2
            # 3er segmento de parent_1
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            
            
            ### HIJO 2 ###

            # 1er segmento de parent_2
            # 2do segmento de parent_1
            # 3er segmento de parent_2
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
        
        
        
        ### SI INDEX DEL PARENT_2 ES ANTES DEL DE PARENT_1
        # ej: parent_1 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        # ej: parent_2 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        else:
            
            first_seg_parent_1 = parent_1[:index_2]
            mid_seg_parent_1 = parent_1[index_2:index_1+1]
            last_seg_parent_1 = parent_1[index_1+1:]
            
            first_seg_parent_2 = parent_2[:index_2]
            mid_seg_parent_2 = parent_2[index_2:index_1+1]
            last_seg_parent_2 = parent_2[index_1+1:]
            
            
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
     
    # cuando no hay crossover
    # si prob_crsvr == 1, siempre hay crossover
    else:
        child_1 = parent_1
        child_2 = parent_2
    
    return child_1,child_2 # cada hijo es un array



############################################################
### MUTANDO LOS HIJOS                                    ###
############################################################

# inputs: child_1, child_2, y la probabilidad de mutacion
# prob mutacion default es 0.2
def mutation(child_1,child_2,prob_mutation=0.2):
    
    # mutado 1
    mutated_child_1 = np.empty((0,len(child_1)))
      
    t = 0 # comenzar con el primer bit o gene
    for i in child_1: # para cada gene 
        
        rand_num_to_mutate_or_not_1 = np.random.rand() # mutar o no???
        
        if rand_num_to_mutate_or_not_1 < prob_mutation:
            
            if child_1[t] == 0: # 0 se convierte en 1
                child_1[t] = 1
            
            else:
                child_1[t] = 0  # 1 se convierte en 0
            
            mutated_child_1 = child_1
            
            t = t+1
        
        else:
            mutated_child_1 = child_1
            
            t = t+1
    
       
    # mutado 2
    mutated_child_2 = np.empty((0,len(child_2)))
    
    t = 0
    for i in child_2:
        
        rand_num_to_mutate_or_not_2 = np.random.rand() # prob. de mutar
        
        if rand_num_to_mutate_or_not_2 < prob_mutation:
            
            if child_2[t] == 0:
                child_2[t] = 1
           
            else:
                child_2[t] = 0
            
            mutated_child_2 = child_2
            
            t = t+1
        
        else:
            mutated_child_2 = child_2
            
            t = t+1
    
    return mutated_child_1,mutated_child_2 # cada hijo mutado es un array
