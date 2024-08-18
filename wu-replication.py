import random
import numpy as np
from scipy.stats import beta
from numpy import random
from scipy.special import comb
from random import sample
from itertools import product
import itertools
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd


###defining vars###
n=20 #agent vector length
k=5 #landscape ruggedness 1 <= K => N-1
total_agents=10 #testing, change to 10
rounds=20 #testing, change to 200
simulation_runs=10 #testing, change to 10000
probability_range=[.4,.6,.8,1] #probability of connection for a one way random network
velocity_range=[1,3,5]
doubles=[]

####Triples for Simultaion Runs###
for probability in probability_range:
	for velocity in velocity_range:
		doubles.append([probability,velocity]) #doubles for variations on simulation runs #proportion
#print(doubles)

######################
###helper fucntions###
######################

#defining completely connected network
#to test, change behavioral_change_best_answer(best_agents,n,k,l,valuation_list,t,average_score_best,complete_network,max_score,velocity)
#and uncomment complete network in simulation code below
def complete(total_agents): #defining completely connected network, to test 
	network=[[1 for agents in range(total_agents)] for other_agents in range(total_agents)]
	return network

#definine one way network
def random_one_way_network(total_agents,probability):
	network=[[0 for agents in range(total_agents)] for other_agents in range(total_agents)]
	for agents in range(total_agents):
		for other_agents in range(total_agents):
			if agents==other_agents:
				network[agents][other_agents]=1
			elif agents!=other_agents:
				if random.uniform(0,1)<probability:
					network[agents][other_agents]=1
	return network

def dfs(visited, G, node, k):
	if node not in visited:
		visited.add(node)
		for n in range(k):
			if G[node][n]==1:
				dfs(visited, G, n, k)

def check_connection(G, k):
	connected=[]
	for n in range(k):
		visited=set()
		dfs(visited, G, n, k)
		connected.append(visited==set(list(range(k))))
	if all(connected):
		return True
	
def maximum_score(n,k,l,valuation_list):
	maximum_score=0
	for i in product([0,1],repeat=n):
		list_mode=list(i)
		score=score_calculation(list_mode,n,k,l,valuation_list)
		if score>maximum_score:
			maximum_score=score
	return maximum_score

def list_split(listA, n):
    for x in range(0, len(listA), n):
        every_chunk = listA[x: n+x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [None for y in range(n-len(every_chunk))]
        yield every_chunk

def score_calculation(string,n,k,l,valuation_list): #calculate fitness score
	total=0
	working_list=[]
	for i in range(n):
		working_list=valuation_list[string[i]][:]
		for index in range(len(l)-1):
			working_list=working_list[string[i-l[index]]][:]
		total+=working_list[string[i-l[len(l)-1]]]
	return (total/n)**8

def list_duplicates_of(seq,item):
	start_at = -1
	locs = []
	while True:
		try:
			loc = seq.index(item,start_at+1)
		except ValueError:
			break
		else:
			locs.append(loc)
			start_at = loc
	return locs

def list_better(seq,value):
	better=[]
	for index in range(len(seq)):
		if seq[index]>value:
			better.append(index)
	return better

# valuation_list=[[random.uniform(0,1) for w in range(2)] for m in range(2**k)]
# 		for i in range(k-1):
# 			valuation_list=list(list_split(valuation_list,2))

#above is the code (located in the "Simulation" block below) which establishes the NK space 
#valuation_list is a list of lists (a list of lists containing three(?) values on a normal dist. from 0-1)

###methods that use valuation_list####
#score_calculation method slices valuation_list to calculate the score at a location

###methods that valuation_list uses###
#list_split establishes the dependencies for all k
#these dependencies are then employed in score_calculation



###search strategy for agents adopting "best" strategy###
#agents always adopt the best strategy of their neighbors 
#rate of social learning varies based on velocity parameter v

def behavioral_change_best_answer(best_agents,n,k,l,valuation_list,t,average_score_best,random_network,max_score,velocity):
	sorting_list=[]

	for agent_index in range(total_agents):
		sorting_list.append(score_calculation(best_agents[agent_index],n,k,l,valuation_list))

	average_score_best[t]+=(((sum(sorting_list)/len(sorting_list))/max_score))

	if t%velocity==0:
		for agent_index in range(total_agents):
			change_number=0
			product=[]
			for num1, num2 in zip(random_network[agent_index], sorting_list):
				product.append(num1 * num2)
			max_index=list_duplicates_of(product, max(product))
			if sorting_list[agent_index]!=max(product):
				best_agents[agent_index]=best_agents[random.choice(max_index)][:]
			else:
				change_number=random.randint(0,n)
				copy_string=best_agents[agent_index][:]
				copy_string[change_number]=(1-copy_string[change_number])
				if score_calculation(copy_string,n,k,l,valuation_list)>max(product):
					best_agents[agent_index]=copy_string[:]
	elif t%velocity!=0:
		for agent_index in range(total_agents):
			change_number=random.randint(0,n)
			copy_string=best_agents[agent_index][:]
			copy_string[change_number]=(1-copy_string[change_number])
			if score_calculation(copy_string,n,k,l,valuation_list)>sorting_list[agent_index]:
				best_agents[agent_index]=copy_string[:]
				

###establishing vars for simulation runs in for loop###
###keeps track of average fitness across rounds###
simulation=0
average_score_best=[0 for s in range(rounds)]

###Simulation runs###
def run(double):
	probability=double[0]
	velocity=double[1]
	simulation=0
	average_score_best=[0 for s in range(rounds)]
	pbar = tqdm(desc="while loop", total=simulation_runs)

	while simulation<simulation_runs:
		t=1
		l=sample(range(1,n),k)
		valuation_list=[[random.uniform(0,1) for w in range(2)] for m in range(2**k)]
		for i in range(k-1):
			valuation_list=list(list_split(valuation_list,2))
		best_agents=[[random.randint(0,2) for w in range(n)] for agents in range(total_agents)]

		#complete_best_agents=best_agents[:]

		random_network=random_one_way_network(total_agents,probability)
		while check_connection(random_network,total_agents)!=True:
			random_network=random_one_way_network(total_agents,probability)

		#complete_network=complete(total_agents)

		max_score=maximum_score(n,k,l,valuation_list)

		while t<rounds:
			behavioral_change_best_answer(best_agents,n,k,l,valuation_list,t,average_score_best,random_network,max_score,velocity)
			t+=1
		simulation+=1
		pbar.update(1)
	pbar.close()
	return [double,average_score_best]

if __name__ == '__main__':
	p=Pool(mp.cpu_count())
	results=p.map(run, doubles)
	df = pd.DataFrame(results)
	df.to_csv('test.csv', index=False, header=False)
    #print(results)
