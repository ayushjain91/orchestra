import numpy
from matplotlib.pylab import *

def get_coclustering_matrix(clusters):
	num_items = len(clusters)
	coclustering_matrix = numpy.zeros((num_items, num_items))
	for i in range(num_items):
		if clusters[i] is None:
			continue
		for j in range(i + 1, num_items):
			if clusters[i] == clusters[j]:
				coclustering_matrix[i][j] = 1
				coclustering_matrix[j][i] = 1
	return coclustering_matrix


def calc_precision(clusters, ground_truth_labels):
	coclustering_matrix = get_coclustering_matrix(clusters)
	gt_coclustering_matrix = get_coclustering_matrix(ground_truth_labels)

	tp = numpy.sum(numpy.logical_and(coclustering_matrix == gt_coclustering_matrix, coclustering_matrix==1))
	p = numpy.sum(coclustering_matrix==1)
	precision = 1.0*tp/p
	return precision

def calc_recall(clusters, ground_truth_labels):
	coclustering_matrix = get_coclustering_matrix(clusters)
	gt_coclustering_matrix = get_coclustering_matrix(ground_truth_labels)

	tp = numpy.sum(numpy.logical_and(coclustering_matrix == gt_coclustering_matrix, coclustering_matrix==1))
	tp_plus_fn = numpy.sum(gt_coclustering_matrix == 1)
	recall = 1.0*tp/tp_plus_fn
	return recall

def calc_accuracy(clusters, ground_truth_labels):
	unique_clusters = list(numpy.unique(clusters))
	try:
		unique_clusters.remove(None)
	except:
		pass
	num_unique_clusters = len(unique_clusters)
	
	unique_ground_truth_labels = list(numpy.unique(ground_truth_labels))
	num_unique_ground_truth_labels = len(unique_ground_truth_labels)

	conf_matrix = numpy.zeros((num_unique_ground_truth_labels, num_unique_clusters))
	for i in range(num_unique_ground_truth_labels):
		for j in range(num_unique_clusters):
			conf_matrix[i][j] = len([_i_ for _i_ in range(len(clusters)) if clusters[_i_] == unique_clusters[j]
				and ground_truth_labels[_i_] == unique_ground_truth_labels[i]])

	errors = 0
	row_idx = 0
	for j in numpy.argmax(conf_matrix, axis=1):
		errors += len([_i_ for _i_ in range(len(clusters)) 
			if ground_truth_labels[_i_] == unique_ground_truth_labels[row_idx]])
		errors -= conf_matrix[row_idx][j]
		row_idx += 1

	accuracy = 1 - 1.0*errors/len(ground_truth_labels)
	return accuracy

def conf_matrix_plot(clusters, ground_truth_labels, savepath, gray=False):
	unique_clusters = list(numpy.unique(clusters))
	try:
		unique_clusters.remove(None)
	except:
		pass
	num_unique_clusters = len(unique_clusters)
	
	unique_ground_truth_labels = list(numpy.unique(ground_truth_labels))
	unique_ground_truth_labels.sort()
	num_unique_ground_truth_labels = len(unique_ground_truth_labels)

	conf_matrix = numpy.zeros((num_unique_ground_truth_labels, num_unique_clusters))
	for i in range(num_unique_ground_truth_labels):
		for j in range(num_unique_clusters):
			conf_matrix[i][j] = len([_i_ for _i_ in range(len(clusters)) if clusters[_i_] == unique_clusters[j]
				and ground_truth_labels[_i_] == unique_ground_truth_labels[i]])

	fig = figure()
	ax = fig.add_subplot(111)

	if gray:
		cax = ax.matshow(conf_matrix, cmap=cm.gray)
	else:
		cax = ax.matshow(conf_matrix)
	cbar = fig.colorbar(cax)
	ax.set_yticks(numpy.arange(num_unique_ground_truth_labels))
	ax.set_yticklabels(unique_ground_truth_labels)
	setp(ax.get_xticklabels(), fontsize=20)
	setp(ax.get_yticklabels(), fontsize=20)
	#xlabel('Assigned Cluster')
	#ylabel('Ground Truth Category')
	cbar.ax.tick_params(labelsize=20) 
	savefig(savepath+'.eps', format='eps', dpi=400)
	savefig(savepath+'.png', format='png', dpi=400)

