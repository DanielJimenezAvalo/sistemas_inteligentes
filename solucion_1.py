
import random
import sys
import operator

class Knapsack(object):    

    #inicializar variables y arrays
    def __init__(self):    

        self.C = 0
        self.weights = []
        self.parents = []
        self.newparents = []
        self.bests = []
        self.best_p = [] 
        self.iterated = 1
        self.population = 0
        self.numiters = 0
        self.champion=[]
        self.champ_val= -1

        # aumentar max recursion para un stack grande
        iMaxStackSize = 15000
        sys.setrecursionlimit(iMaxStackSize)

    # crear poblacion inicial 
    def initialize(self):

        for i in range(self.population):
            parent = []
            for k in range(0, len(self.weights)):
                k = random.randint(0, 1)
                parent.append(k)
            self.parents.append(parent)

    # setear propiedades del problema
    def properties(self, weights, C, population, num_iters):

        self.weights = weights
        self.C = C
        self.population = population
        self.numiters= num_iters
        self.initialize()

    def FO(self, x):
        return (x[2] or x[3] or x[4]) and\
               (x[2] or not x[6] or x[7]) and\
               (x[0] or x[4] or x[6]) and\
               (x[2] or x[6]) and\
               (not x[0] or x[1] or not x[2]) and\
               (x[0] or not x[4]) and\
               (x[4] or x[5]) and\
               (x[1] or x[3]) and\
               (not x[0] or not x[1]) and\
               (not x[5] or not x[6])
    
    def fitness(self, item):
        f = self.FO(item)
        if f == self.C: #si f es TRUE
            return f
        return -1
    
    # correr varias generaciones del algoritmo
    def evaluation(self):

        # loop en los padres, calcula fitness
        best_pop = self.population // 2
        for i in range(len(self.parents)):
            parent = self.parents[i]
            ft = self.fitness(parent)
            self.bests.append((ft, parent))

        # ordenar por fitness        
        self.bests.sort(key=operator.itemgetter(0), reverse=True)
        self.best_p = self.bests[:best_pop]
        self.best_p = [x[1] for x in self.best_p]

    # mutatar hijos bajo una condicion
    def mutation(self, ch):

        for i in range(len(ch)):        
            k = random.uniform(0, 1)
            if k < 0.1:
                # arriba es la prob. de mutacion
                # mutacion: 0 a 1, 1 a 0
                if ch[i] == 1:
                    ch[i] = 0
                else: 
                    ch[i] = 1
        return ch

    # crossover 
    def crossover(self, ch1, ch2):

        threshold = random.randint(1, len(ch1)-1)
        tmp1 = ch1[threshold:]
        tmp2 = ch2[threshold:]
        ch1 = ch1[:threshold]
        ch2 = ch2[:threshold]
        ch1.extend(tmp2)
        ch2.extend(tmp1)

        return ch1, ch2

    # correr el algoritmo
    def run(self):
        for n_it in range (1,self.numiters+1):        
            self.evaluation()
            newparents = []
            pop = len(self.best_p)

            # crear array con enteros sin repeticion
            sample = random.sample(range(pop), pop)
            for i in range(0, pop):
                # index aleatorio de mejores hijos
                if i < pop-1:
                    r1 = self.best_p[i]
                    r2 = self.best_p[i+1]
                    nchild1, nchild2 = self.crossover(r1, r2)
                    newparents.append(nchild1)
                    newparents.append(nchild2)
                else:
                    r1 = self.best_p[i]
                    r2 = self.best_p[0]
                    nchild1, nchild2 = self.crossover(r1, r2)
                    newparents.append(nchild1)
                    newparents.append(nchild2)

            # mutar hijos, padres
            for i in range(len(newparents)):
                newparents[i] = self.mutation(newparents[i])

            self.iterated += 1
            if self.bests[0][0]>self.champ_val:
                self.champion =self.bests[0][1]
                self.champ_val=self.bests[0][0]
            print("Iter.%3d " %n_it, end="")
            print(self.bests[0:3])
            self.parents = newparents
            self.bests = []
            self.best_p = []
        print("Mejor de todos:",self.champ_val,self.champion)


n = 8
# valor random de las variables
weights = [random.randint(0, 1) for i in range(n)]

#valor de la FO
C = True
population = 20
num_iters=50

k = Knapsack()
k.properties(weights, C, population, num_iters)
k.run()