import matplotlib.pyplot as plt
from matplotlib import rcParams,rc
import statistics
from scipy.linalg import norm
from ggplot import *
import statistics as stats 

rcParams['font.family'] = 'STIXGeneral'
rcParams['font.sans-serif'] = ['STIXGeneral']
rcParams['font.size']= 14

rc('text', usetex=True)

font = {'fontname':'STIXGeneral'}

def draw_box(x5,x10,x15):
	plt.boxplot([X5,X10,X15],notch=False,meanline=True,showmeans=True)
	plt.ylabel('Proportion of correct predictions')
	plt.xlabel('No. of IP addresses assigned by target server')
	plt.xticks([1,2,3], [5,10,15])
	plt.title('Performance of Algorithm 1')
	plt.savefig('barplot.pdf',format='pdf')


def draw_stack(X,L,filename='predacc.pdf'):
	P =[]
	for x in X: 
		P.append(plt.bar([i for i in range(len(x))], x))

	plt.ylabel('Proportion of predictions relative to number of predictions',wrap=True)
	plt.xlabel('Number of IP addresses correctly predicted', wrap=True)
	plt.title('Performance of predicting leading 16 bits', wrap=True)
	plt.xticks([i for i in range(len(X[-1]))], [i for i in range(len(X[-1]))])
	plt.grid(True,linewidth=0.2,alpha=0.5)
	plt.legend(P,L)
	plt.savefig(filename,format='pdf')


def draw_K_variation(X,y,z,a,filename='kvar.pdf'):
	l1norm = norm(a)
	A=[round(v/l1norm) for v in a]
	locs = [i for i in range(len(X))]
	plt.scatter(locs,y)
	for i,v in enumerate(X): 
		plt.annotate(v, (locs[i],y[i]))
	plt.grid(True,linewidth=0.2,alpha=0.5)
	plt.xticks(locs,z)
	plt.xlabel('Final value of K')
	plt.ylabel(r"Proportion of correct predictions for $>\lceil N/2 \rceil$ values")
	plt.savefig(filename,format='pdf')

def draw_bar(X,L,filename='bar.pdf'):
	P=[]
	# plt.rc('legend', fontsize=12)
	# plt.rc('xtick', labelsize=18)
	# plt.rc('ytick', labelsize=18)
	# plt.rc('figure', titlesize=18)

	P.append(plt.bar([0,1.1],X1, width=0.5))
	P.append(plt.bar([0.5,1.6],X1M, width=0.5))
	

	plt.xticks([0.25,1.25],['0/1','1/1'])
	plt.ylabel('Proportion of predictions relative to number of predictions', wrap=True)
	plt.xlabel('Number of IP addresses correctly predicted (N=1)', wrap=True)
	plt.title('Performance of predicting leading 16 bits', wrap=True)
	plt.grid(True,linewidth=0.2,alpha=0.5)
	plt.legend(P,L)

	plt.savefig(filename,format='pdf')

def scatter_plot(clusters,dataset,N,score,bytes=[1,2]):
	X = []
	Y = [] 
	C = [] #Cluster labels (accuracy values)
	i = 0 
	cid = 0
	datasetname = dataset #dataset[5:-4].replace('_', ' ')
	for cluster in clusters:
		for assignment in cluster.assignments:
			X.append(i)
			Y.append(stats.mean(assignment.values)) 
			C.append(cid)
			i+=1 
		cid+=1 
	L1norm = norm(Y)
	Y = [round(y/L1norm,3) for y in Y]
	
	plt.scatter(X, Y, c=C, marker='+')

	title = 'Clusters for '+datasetname
	title += "(N="+str(N)+", "
	title += 'M='+str(len(X))+', '
	title += 'W='+str(score)+')'

	plt.title(title)
	plt.xlabel("Indexes of data points")
	plt.ylabel("Normalized numeric values for bytes "+str(bytes))
	plt.grid(True)
	#plt.legend()
	plt.savefig('clusters_scattered_'+str(N)+'_'+dataset[5:-4]+'_'+str(bytes[0])+str(bytes[1])+'.pdf',format='pdf')



X1  = [0.46116970278044106, 0.538830297219559]
X5  = [0.0, 0.0035971223021582736, 0.04316546762589928, 0.2446043165467626, 0.3669064748201439, 0.34172661870503596]
X10 = [0.0, 0.0, 0.0, 0.007194244604316547, 0.0, 0.007194244604316547, 0.02158273381294964, 0.10071942446043165, 0.19424460431654678, 0.3237410071942446, 0.34532374100719426]
X15 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06521739130434782, 0.13043478260869565, 0.32608695652173914, 0.4782608695652174]
X1M = [0.8536585365853658, 0.14634146341463414]

# for x in [X1,X5,X10,X15,X1M]:
# 	print(statistics.mean(x), statistics.stdev(x), max(x))

#draw_stack([X5,X10,X15],['N=5','N=10','N=15'],filename='accuracy-various-N.pdf')
#draw_bar([X1,X1M],['Clustering of Algorithm 1','Uniform Predictor'],filename='cluster-rand-compare.pdf')

X=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.5,0.55,1.0]
y=[0.558, 0.59, 0.572, 0.565, 0.574, 0.574, 0.581, 0.616, 0.608,0.6439,0.6585,0.696]
z=[4, 13, 31, 51, 64, 76, 86, 103, 103,112,117,132]
a=[43, 22, 19, 22, 24, 26, 34, 37, 40,43,47]
#draw_K_variation(X,y,z,a)
#scatter_clusters()

