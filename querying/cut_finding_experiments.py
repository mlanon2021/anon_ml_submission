import graph_tool as gt
import matplotlib.pyplot as plt
import numpy as np
from graph_tool.topology import shortest_distance

from other_querying_strategies.s2_querying import s2, random_not_s2
from prediction_strategies.labelpropgation import active_label_prop_dirty
from querying.evaluate_queries import cut_finding_performance
from querying.greedy_ray_maximizer import binary_greedy_ray_maximizer, binary_cal_style
from read_graphs.read import build_graph
from synthetic_data_generation.sbm_generation import generate_lattice

# ssns.set()
# sns.palplot(sns.color_palette("cubehelix", 5))
# matplotlib.pyplot.set_cmap("cubehelix")

font = {'family': 'normal',
        'size': 24}

plt.rc('font', **font)


def plot(cut_finding_s2, cut_finding_ray, cut_finding_cal, cut_finding_random, cut_finding_lp, ax, labels=True,
         y_text="", dataset_name=""):
    cmap = plt.get_cmap("viridis")
    indices = np.linspace(0, cmap.N, 10)
    my_colors = [cmap(int(i)) for i in indices]
    y_max = np.max(np.concatenate((cut_finding_s2, cut_finding_ray, cut_finding_cal, cut_finding_random)))
    num = 20
    y = np.ones(num)
    if y_text != "":
        ax.set_ylabel(y_text)

    if labels:
        y[:len(cut_finding_ray)] = cut_finding_ray / y_max
        ax.plot(y, linestyle="-", marker="s", color=my_colors[0], label="greedy")

        y = np.ones(num)
        y[:len(cut_finding_cal)] = cut_finding_cal / y_max
        ax.plot(y, linestyle="--", marker="d", color=my_colors[2], label="selective sampling")

        y = np.ones(num)
        y[:len(cut_finding_s2)] = cut_finding_s2 / y_max
        ax.plot(y, linestyle=":", marker="v", color=my_colors[4], label="$\mathregular{S^2}$")

        y = np.ones(num)
        y[:len(cut_finding_lp)] = cut_finding_lp / y_max
        ax.plot(y, linestyle="--", marker="^", color=my_colors[6], label="active lp")

        y = np.ones(num)
        y[:len(cut_finding_random)] = cut_finding_random / y_max
        ax.plot(y, linestyle="-.", marker="o", color=my_colors[8], label="random")

    else:
        y[:len(cut_finding_ray)] = cut_finding_ray[:20] / y_max
        ax.plot(y, linestyle="-", color=my_colors[0], marker="s")

        y = np.ones(num)
        y[:len(cut_finding_cal)] = cut_finding_cal / y_max
        ax.plot(y, linestyle="--", color=my_colors[2], marker="d")

        y = np.ones(num)
        y[:len(cut_finding_s2)] = cut_finding_s2 / y_max
        ax.plot(y, linestyle=":", color=my_colors[4], marker="v")

        y = np.ones(num)
        y[:len(cut_finding_lp)] = cut_finding_lp / y_max
        ax.plot(y, linestyle="--", color=my_colors[6], marker="^")

        y = np.ones(num)
        y[:len(cut_finding_random)] = cut_finding_random / y_max
        ax.plot(y, linestyle="-.", color=my_colors[8], marker="o")

    ax.set_xlim([0, 20])

    ax.set_xlabel(dataset_name)
    ax.xaxis.set_ticks([0, 5, 10, 15, 20])
    ax.yaxis.set_ticks([0, 0.5, 1.0])

def test_with_error_bars():
    gt.seed_rng(444)
    np.random.seed(444)


    num_runs = 10
    for dataset in range(4):
        print("==================================")
        print("dataset",dataset)
        #g, labels = generate_lattice(2,4)
        if dataset == 0:
            g, labels, _, _ = build_graph("moons")
        elif dataset == 1:
            g, labels, _, _ = build_graph("iris")
        elif dataset == 2:
            g, labels = generate_lattice(20, 2)
        else:
            g, labels = generate_lattice(2, 10)
        print("===================================")

        accs_rays = []
        accs_cal = []
        accs_random = []
        accs_s2 = []
        accs_lp = []

        cut_finding_ray = []
        cut_finding_cal = []
        cut_finding_random = []
        cut_finding_s2 = []
        cut_finding_lp = []

        for z in range(num_runs):
            print("==================================")
            print(z)
            print("===================================")


            starting_vertex = np.random.choice(np.where(labels)[0], size=1)[0] #always pick the same class s.t. the first accuracy is always the same
            print("starting_vertex", starting_vertex, labels[starting_vertex], np.average(labels))
            distances = shortest_distance(g)
            distance_matrix = distances.get_2d_array(range(g.num_vertices()))
            # return
            print("============================ray=================================")
            # preds = active_label_prop(g, labels, distance_matrix)
            queries_ray, acc_ray = binary_greedy_ray_maximizer(g, labels, dists=distance_matrix,
                                                               starting_vertex=starting_vertex)
            accs_rays.append(acc_ray)
            cut_finding_ray.append(cut_finding_performance(g, labels, queries_ray))
            print("===============================cal==============================")

            queries_cal, acc_cal = binary_cal_style(g, labels, dists=distance_matrix, starting_vertex=starting_vertex)
            accs_cal.append(acc_cal)
            cut_finding_cal.append(cut_finding_performance(g, labels, queries_cal))
            print("===============================random==============================")
            weight_prop = g.new_edge_property("int", val=1)
            queries_random, acc_random = random_not_s2(g, weight_prop, labels=labels, budget=20,
                                                       starting_vertex=starting_vertex)
            accs_random.append(acc_random)
            cut_finding_random.append(cut_finding_performance(g, labels, queries_random))
            print("=================================s2============================")
            queries_s2, acc_s2 = s2(g, weight_prop, labels=labels, budget=20, starting_vertex=starting_vertex)
            accs_s2.append(acc_s2)
            cut_finding_s2.append(cut_finding_performance(g, labels, queries_s2))
            print("==================================alp===========================")
            queries_lp, acc_lp = active_label_prop_dirty(g, weight_prop, labels=labels, budget=20,
                                                         starting_vertex=starting_vertex)
            accs_lp.append(acc_lp)
            cut_finding_lp.append(cut_finding_performance(g, labels, queries_lp))
            #if i == 0:
            #    y_text = "% found cut vertices"
            #else:
            #    y_text = ""
            #plot(cut_finding_s2, cut_finding_ray, cut_finding_cal, cut_finding_random, cut_finding_lp, axes[0, i], i == 0,
            #     y_text)

        cmap = plt.get_cmap("viridis")
        indices = np.linspace(0, cmap.N, 10)
        my_colors = [cmap(int(i)) for i in indices]
        num = 20

        linestyles=["-","--",":","--","-."]
        markers=["s","d","v","^","o"]
        labels=["greedy", "selective sampling", "active lp", "$\mathregular{S^2}$", "random"]

        plt.figure(figsize=(20,8), dpi=400)

        for m, method in enumerate([accs_rays, accs_cal, accs_random, accs_s2, accs_lp]):

            temp = np.ones((num_runs,20))
            for i,acc in enumerate(method):
                temp[i,:len(acc)] = acc
                temp[i,len(acc)-1:] = acc[-1]

            method = temp

            method_avg = np.average(method, axis=0)

            plt.errorbar(np.arange(1,1+method_avg.size)+m*0.09, method_avg, yerr=[method_avg-np.quantile(method, q=.10, axis=0), -method_avg+np.quantile(method, q=.90, axis=0)],
                         linestyle="none", color=my_colors[2*m], marker=markers[m], label=labels[m], elinewidth=2)


        plt.legend(loc="lower right")
        plt.legend(loc="lower right")
        plt.xticks(range(1, 21))
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylabel("accuracy")
        plt.xlabel("queries")

        plt.savefig("acc_"+str(dataset)+".png", bbox_inches='tight', dpi=400)
        #plt.show()

        plt.figure(figsize=(20, 8), dpi=400)

        y_max = 0
        for method in [cut_finding_ray, cut_finding_cal, cut_finding_random, cut_finding_s2, cut_finding_lp]:
            for attempt in method:
                temp_max = np.max(attempt)
                if temp_max > y_max:
                    y_max = temp_max

        for m, method in enumerate([cut_finding_ray, cut_finding_cal, cut_finding_random, cut_finding_s2, cut_finding_lp]):

            temp = np.ones((num_runs, 20))
            for i, acc in enumerate(method):
                temp[i, :acc.size] = acc/y_max
                temp[i, acc.size - 1:] = acc[-1]/y_max

            method = temp

            method_avg = np.average(method, axis=0)

            plt.errorbar(np.arange(1, 1 + method_avg.size) + m * 0.09, method_avg,
                         yerr=[method_avg - np.quantile(method, q=.10, axis=0),
                               -method_avg + np.quantile(method, q=.90, axis=0)],
                         linestyle="none", color=my_colors[2 * m], marker=markers[m], label=labels[m])

        plt.legend(loc="lower right")
        plt.xticks(range(1, 21))
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
        plt.ylabel("% of found cut vertices")
        plt.xlabel("queries")
        plt.savefig("cut_finding_"+str(dataset)+".png", bbox_inches='tight', dpi=400)

if __name__ == "__main__":
    test_with_error_bars()
