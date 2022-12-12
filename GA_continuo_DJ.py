# -*- coding: latin-1 -*-
### FUNCION DE HIMMELBLAU 
### MIN ((x**2)+y-11)**2+(x+(y**2)-7)**2

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import GA_Functions_Continuous_DJ as genf
import time


# hiperparametros (los pone el usuario)
prob_crsvr = 1 # probab. de crossover
prob_mutation = 0.3 # probab. de mutacion
population = 120 # tamaño de poblacion
generations = 80 # numero de generaciones


# encoding de las variables x, y
# 13 genes para x y 13 genes para y (arbitrario, precision)
x_y_string = np.array([0,1,0,0,0,1,0,0,1,0,1,1,1,
                       0,1,1,1,0,0,1,0,1,1,0,1,1]) # initial solution


# array vacio para poner poblacion inicial
pool_of_solutions = np.empty((0,len(x_y_string)))


# array vacio para guardar la mejor solucion de 
# cada generacion
# para luego plotear la convergencia del algoritmo
best_of_a_generation = np.empty((0,len(x_y_string)+1))


# mezclar al azar los elementos de la solucion inicial (vector)
# n veces, donde n= num. de la poblacion
for i in range(population):
    rd.shuffle(x_y_string)
    pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))


# ahora pool_of_solutions, tiene n individuos


start_time = time.time() # tiempo inicial

gen = 1 # empezar en generacion no.1 


for i in range(generations): # n (generaciones) veces
    
    # array vacio para la nueva generacion
    new_population = np.empty((0,len(x_y_string)))
    
    # array vacio para la nuebva generacion +  valor funcion objetivo
    new_population_with_obj_val = np.empty((0,len(x_y_string)+1))
    
    # array vacio para grabar la mejor solucion (cromosoma)
    # de cada generacion
    sorted_best_for_plotting = np.empty((0,len(x_y_string)+1))
    
    print()
    print()
    print("--> Generacion: #", gen) # tracking purposes
    
    
    family = 1
    
    
    for j in range(int(population/2)): # population/2 por que hay 2 padres
        
        print()
        print("--> Familia: #", family) # tracking
        
            
        # escoger 2 padres usando torneo
        # "genf.find_parents_ts"[0] da parent_1
        # "genf.find_parents_ts"[1] da parent_2
        parent_1 = genf.find_parents_ts(pool_of_solutions,gen)[0]
        parent_2 = genf.find_parents_ts(pool_of_solutions,gen)[1]
        
        
        # crossover de los 2 padres para obtener 2 hijos
        # "genf.crossover"[0] da child_1
        # "genf.crossover"[1] da child_2
        child_1 = genf.crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[0]
        child_2 = genf.crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[1]
        
        
        # mutacion de los 2 hijos para dar 2 hijos mutados
        # "genf.mutation"[0] da mutated_child_1
        # "genf.mutation"[1] da mutated_child_2
        mutated_child_1 = genf.mutation(child_1,child_2,
                                      prob_mutation=prob_mutation)[0]  
        mutated_child_2 = genf.mutation(child_1,child_2,
                                      prob_mutation=prob_mutation)[1] 
        
        
        # obtener la func objetivo (fitness) para los 2 hijos mutados
        # "genf.objective_value"[2] da func obj del hijo mutado
        obj_val_mutated_child_1 = genf.objective_value(mutated_child_1,gen)[2]
        obj_val_mutated_child_2 = genf.objective_value(mutated_child_2,gen)[2]
        
        
        
        # mostrar hijos mutados y func objetivo
        print()
        print("Val F.Ob. Hijo Mutado #1 en Generacion #{} : {}".
              format(gen,obj_val_mutated_child_1))
        print("Val F.Ob. Hijo Mutado #2 en Generacion #{} : {}".
              format(gen,obj_val_mutated_child_2))
        
        if genf.objective_value(mutated_child_2,gen)[4]>0:
            print("------violaciones de restriccion------")
            print("val F.Ob sin penalizacion en la generacion #{} : {}".
                  format(gen,genf.objective_value(mutated_child_2,gen)[3]))
            print("val de la penalizacion en la generacion #{} : {}".
                  format(gen,genf.objective_value(mutated_child_2,gen)[4]))

        # para cada hijo mutado, juntarle su func obj
        mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1,
                                               mutated_child_1)) # lines 103 and 111
        
        mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2,
                                               mutated_child_2)) # lines 105 and 112
        
        
        # craer nueva poblacion para sig. generacion
        # para cada familia, obtenemos 2 soluciones
        # las vamos añadiendo hasta completar la poblacion
        # al final, deberiamos tener el mismo tamaño de poblacion
        new_population = np.vstack((new_population,
                                    mutated_child_1,
                                    mutated_child_2))
        
        
        # incluir func obj
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                 mutant_1_with_obj_val,
                                                 mutant_2_with_obj_val))
        
        
        family = family+1
        
          
    # reemplazar poblacion actual con poblacion nueva (sgte generacion)
    # que viene a ser la poblac inicial de la sgte generacion
    pool_of_solutions = new_population
    
    
    # para cada generacion
    # hallar su mejor solucion
    # asi que ordenamos por el index [0], que es el fitness
    sorted_best_for_plotting = np.array(sorted(new_population_with_obj_val,
                                               key=lambda x:x[0]))
    
    
    best_of_a_generation = np.vstack((best_of_a_generation,
                                      sorted_best_for_plotting[0]))
    
    
    # aumentar el contador de generaciones
    gen = gen+1       



end_time = time.time() # tiempo final


# proceso de la ultima generacion
sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                         key=lambda x:x[0]))

sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                         key=lambda x:x[0]))


best_string_convergence = sorted_last_population[0]

best_string_overall = sorted_best_of_a_generation[0]


print()
print()
print("------------------------------")
print()
print("Tiempo ejecucion en segs:",end_time - start_time) # exec. time
print()
print("Solucion Final (Convergencia):",best_string_convergence[1:]) # final solution entire chromosome
print("X codificado (Convergencia):",best_string_convergence[1:14]) # final solution x chromosome
print("Y codificado (Convergencia):",best_string_convergence[14:]) # final solution y chromosome
print()
print("Solucion final (mejor):",best_string_overall[1:]) # final solution entire chromosome
print("X codificado (mejor):",best_string_overall[1:14]) # final solution x chromosome
print("Y codificado (mejor):",best_string_overall[14:]) # final solution y chromosome

# para decodificar x , y a sus valores reales equivalentes
# la funcion "genf.objective_value" devuelve 3 datos -->
# [0] = x 
# [1] = y 
# [2] = func obj
# usamos "best_string[1:]" por que no nos interesa el elemento 0
# que es la func obj
final_solution_convergence = genf.objective_value(best_string_convergence[1:],gen)

final_solution_overall = genf.objective_value(best_string_overall[1:],gen)



print()
print("X real (Convergencia):",round(final_solution_convergence[0],5)) # real value of x
print("Y real (Convergencia):",round(final_solution_convergence[1],5)) # real value of y
print("Func Obj - Convergencia:",round(final_solution_convergence[2],5)) # obj val of final chromosome
print()
print("X real (mejor):",round(final_solution_overall[0],5)) # real value of x
print("Y real (mejor):",round(final_solution_overall[1],5)) # real value of y
print("Func Obj - mejor en la corrida:",round(final_solution_overall[2],5)) # obj val of final chromosome
print()
print("------------------------------")


### Plotear mejor solucion por generacion ###

best_obj_val_convergence = (best_string_convergence[0]) 

best_obj_val_overall = best_string_overall[0]


plt.plot(best_of_a_generation[:,0]) 
plt.axhline(y=best_obj_val_convergence,color='r',linestyle='--')

plt.axhline(y=best_obj_val_overall,color='r',linestyle='--')

plt.title("Z a lo largo de las Generaciones",fontsize=20,fontweight='bold')
plt.xlabel("Generación",fontsize=18,fontweight='bold')
plt.ylabel("Z",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')


if sorted_best_of_a_generation[-1][0] > 2:
    k = 0.8
elif sorted_best_of_a_generation[-1][0] > 1:
    k = 0.5
elif sorted_best_of_a_generation[-1][0] > 0.5:
    k = 0.3
elif sorted_best_of_a_generation[-1][0] > 0.3:
    k = 0.2
else:
    k = 0.1

xyz1 = (generations/2.4,best_obj_val_convergence)
xyzz1 = (generations/2.2,best_obj_val_convergence+k)

plt.annotate("En Convergencia: %0.5f" % best_obj_val_convergence,xy=xyz1,xytext=xyzz1,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')


xyz2 = (generations/6,best_obj_val_overall)
xyzz2 = (generations/5.4,best_obj_val_overall+(k/2))

plt.annotate("Mínimo final: %0.5f" % best_obj_val_overall,xy=xyz2,xytext=xyzz2,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')


plt.show()
