import random
import math

def build_adjacency_list():
    adjacency_list = {}
    num_vertices, num_edges = map(int,input("Enter the number of vertices and edges : ").split())
    
    for i in range(num_edges):
        node1, node2 = map(int, input().split())
        if node1 != node2 and node2 not in adjacency_list.get(node1, []):
            adjacency_list.setdefault(node1,[]).append(node2)
            adjacency_list.setdefault(node2,[]).append(node1)
    
    # for node,neighbor in adjacency_list.items():
    #     print(f"{node} : {', '.join(map(str, neighbor))}")
    
    return adjacency_list
        

#Heuristic-  1
# first you have to choose one unlabed vertex and choose that vertex neighbor randomly (one of them) and give both vertex
# to label 1 and that both vertex common neighbor give label 0  but in case if u choose vertex which has no unlabeled vertex then 
# you have to check three condition 
# 1 . if that selected vertex neighbor has sum of label is greater or than or equal to 2 then give that selected vertex to label 0
# 2. if first condition is not means neigbor sum is <2 then make that neighbor label 1 and that selected vertex label is also 
# assign to 1

def Heuristic_1(adjacency_list):
    labels = {vertex: -1 for vertex in adjacency_list.keys()}

    def get_neighbors_with_label(vertex, label):
        return [neighbor for neighbor in adjacency_list[vertex] if labels[neighbor] == label]

    def get_unlabeled_vertices():
        return [vertex for vertex, label in labels.items() if label == -1]

    def label_vertex_and_neighbors(vertex):
        labels[vertex] = 1
        neighbors = adjacency_list[vertex]
        random_neighbor = random.choice(neighbors)
        labels[random_neighbor] = 1

        common_neighbors = set(neighbors).intersection(adjacency_list[random_neighbor])
        for common_neighbor in common_neighbors:
            labels[common_neighbor] = 0

    def has_two_labeled_neighbors(vertex):
        labeled_neighbors = get_neighbors_with_label(vertex, 1)
        return len(labeled_neighbors) >= 2

    def sum_labels_of_neighbors(vertex):
        return sum(labels[neighbor] for neighbor in adjacency_list[vertex])

    while -1 in labels.values():
        unlabeled_vertices = get_unlabeled_vertices()
        start_vertex = random.choice(unlabeled_vertices)

        if any(labels[neighbor] == -1 for neighbor in adjacency_list[start_vertex]):
            label_vertex_and_neighbors(start_vertex)
        else:
            neighbor_sum = sum_labels_of_neighbors(start_vertex)
            if neighbor_sum >= 2:
                labels[start_vertex] = 0
            else:
                for neighbor in adjacency_list[start_vertex]:
                    # if labels[neighbor] == -1:
                    labels[neighbor] = 1
                    break
                labels[start_vertex] = 1

    return labels
    


#Heuristic-  2
# select a random vertex and give that vertex to label 2 and give label 1 to one neighbor of that selected vertex
# and give 0 to all other remaining vertex and in case if you select a vertex which has no unlabeled vertex then
# give that vertex to label 1 and one of that neighbor give label 1 

def Heuristic_2(adjacency_list):
    labels = {vertex: -1 for vertex in adjacency_list.keys()}

    def get_unlabeled_vertices():
        return [vertex for vertex, label in labels.items() if label == -1]

    def label_vertex_and_neighbors(vertex):
        labels[vertex] = 2
        neighbors = adjacency_list[vertex]
        if any(labels[neighbor] == -1 for neighbor in neighbors):
            unlabeled_neighbors = [neighbor for neighbor in neighbors if labels[neighbor] == -1]
            chosen_neighbor = random.choice(unlabeled_neighbors)
            labels[chosen_neighbor] = 1
            for neighbor in unlabeled_neighbors:
                if neighbor != chosen_neighbor:
                    labels[neighbor] = 0
        else:
            # If all neighbors are labeled, assign label 1 to the vertex and one neighbor
            labels[vertex] = 1
            chosen_neighbor = random.choice(neighbors)
            labels[chosen_neighbor] = 1

    while -1 in labels.values():
        unlabeled_vertices = get_unlabeled_vertices()
        start_vertex = random.choice(unlabeled_vertices)
        label_vertex_and_neighbors(start_vertex)

    return labels




#Heuristic-  3

def Heuristic_3(adjacency_list):
    labels = {vertex: -1 for vertex in adjacency_list.keys()}

    def get_unlabeled_vertices():
        return [vertex for vertex, label in labels.items() if label == -1]

    def find_max_degree_vertex(unlabeled_vertices):
        max_degree_vertex = None
        max_degree = -1
        for vertex in unlabeled_vertices:
            degree =0
            for neigh in adjacency_list[vertex]:
                if labels[neigh] == -1:
                    degree = degree+1
            # degree = len(adjacency_list[vertex])
            if degree > max_degree:
                max_degree = degree
                max_degree_vertex = vertex
        return max_degree_vertex

    def label_vertex_and_neighbors(vertex):
        labels[vertex] = 2
        neighbors = adjacency_list[vertex]
        if any(labels[neighbor] == -1 for neighbor in neighbors):
            unlabeled_neighbors = [neighbor for neighbor in neighbors if labels[neighbor] == -1]
            chosen_neighbor = random.choice(unlabeled_neighbors)
            labels[chosen_neighbor] = 1
            print("chosen_neighbor: ", chosen_neighbor)
            for neighbor in unlabeled_neighbors:
                if neighbor != chosen_neighbor:
                    if labels[neighbor] == -1:
                        #  print(labels[neighbor])
                         labels[neighbor] = 0
        else:
            # If all neighbors are labeled, assign label 1 to the vertex and one neighbor
            labels[vertex] = 1
            chosen_neighbor = random.choice(neighbors)
            print("chosen_neighbor: ", chosen_neighbor)
            labels[chosen_neighbor] = 1

    while -1 in labels.values():
        unlabeled_vertices = get_unlabeled_vertices()
        # Find maximum degree vertex among the unlabeled vertices
        max_degree_vertex = find_max_degree_vertex(unlabeled_vertices)
        print(max_degree_vertex)
        label_vertex_and_neighbors(max_degree_vertex)

    return labels



def check_feasible(vertex_labels, adjacency_list):
    # print("check")
    for vertex, label in vertex_labels.items():
        # print(vertex, "->", label)
        if label == 0:
            neighbor_sum = sum(vertex_labels[neighbor] for neighbor in adjacency_list[vertex])
            # print(neighbor_sum)
            if neighbor_sum < 2:
                return False
        elif label == 1:
            neighbor_sum = sum(vertex_labels[neighbor] for neighbor in adjacency_list[vertex])
            if neighbor_sum < 1:
                return False
        elif label == 2:
            neighbor_sum = sum(vertex_labels[neighbor] for neighbor in adjacency_list[vertex])
            if neighbor_sum < 1:
                return False
                
    return True
    
    

def make_feasible(vertex_labels, adjacency_list):
    # print("Before Modification:", vertex_labels)
    
    labels = dict(vertex_labels)
    for vertex, label in labels.items():
        if label == 0:
            neighbor_sum = sum(labels[neighbor] for neighbor in adjacency_list[vertex])
            # print("ns: ",neighbor_sum)
            if neighbor_sum >= 2:
                continue
            elif neighbor_sum == 1:
                # Find the neighbor labeled 1 and change it to 2
                for neighbor in adjacency_list[vertex]:
                    if labels[neighbor] == 1:
                        labels[neighbor] = 2
                        # print("**")
                        break
            elif neighbor_sum == 0:
                # If neighbor_sum is 0, make any two neighbors label 1
                count = 0
                if len(adjacency_list[vertex]) >= 2:
                    for neighbor in adjacency_list[vertex]:
                        if count < 2:
                            labels[neighbor] = 1
                            count += 1
                else:
                    for neighbor in adjacency_list[vertex]:
                        labels[neighbor] = 2

        else:
            neighbor_sum = sum(labels[neighbor] for neighbor in adjacency_list[vertex])
            if neighbor_sum >= 1:
                continue
            else:
                for neighbor in adjacency_list[vertex]:
                    labels[neighbor] = 1
                    # print("**")
                    break
    # print("After Modification:", labels)
    return labels



def cost_function(labels):
    return sum(labels.values())
    

def generate_neighbor(labels, adjacency_list):
    # neighbor_labels = labels.copy()  # Create a copy of the current solution
    # vertex_to_flip = random.choice(list(labels.keys()))  # Randomly select a vertex
    # neighbor_labels[vertex_to_flip] = 0  # Flip the label
    # # Check feasibility and adjust labels if necessary
    # if not check_feasible(adjacency_list, neighbor_labels):
    #     neighbor_labels = make_feasible(adjacency_list, neighbor_labels)
        
    # return neighbor_labels
    
    
   

    
    # Swap labels of two random vertices
    neighbor_labels = labels.copy()
    vertex1 = random.choice(list(labels.keys()))
    vertex2 = random.choice(list(labels.keys()))
    neighbor_labels[vertex1] = 0
    neighbor_labels[vertex2] = 0
    print("before feasible: " ,neighbor_labels)
    print("cost : ", sum(neighbor_labels.values()))
    # Check feasibility and adjust labels if necessary
    if not check_feasible(neighbor_labels, adjacency_list):
        neighbor_labels = make_feasible(neighbor_labels, adjacency_list)
    
    print("after feasible: " ,neighbor_labels)
    print("cost : ", sum(neighbor_labels.values()))
    print()
    return neighbor_labels


def generate_neighbors(current_solution, adjacency_list, number_of_neighbors):
    neighbors = []
    for _ in range(number_of_neighbors):
        neighbor_labels = generate_neighbor(current_solution, adjacency_list)
        neighbor_cost = cost_function(neighbor_labels)
        neighbors.append((neighbor_labels, neighbor_cost))

    return neighbors



def simulated_annealing(adjacency_list, initial_temperature, cooling_rate, max_iterations):
    number_of_neighbors = 5
    best_solution = None
    best_cost = float('inf')

    global accepted_solutions
    accepted_solutions = []

    itr = 5

    for i in range(itr):
        current_labels = Heuristic_1(adjacency_list)
        current_cost = cost_function(current_labels)

        if current_cost < best_cost:
            best_solution = current_labels
            best_cost = current_cost

    #accepted_solutions.append((best_solution, best_cost))

    # for i in range(itr):
    #     current_labels = Heuristic_2(adjacency_list)
    #     current_cost = cost_function(current_labels)

    #     if current_cost < best_cost:
    #         best_solution = current_labels
    #         best_cost = current_cost

    accepted_solutions.append((best_solution, best_cost))

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate 5 neighbors
        neighbors = generate_neighbors(best_solution, adjacency_list, number_of_neighbors)

        if neighbors:
            # Select the best neighbor
            best_neighbor, best_neighbor_cost = min(neighbors, key=lambda x: x[1])

            # If the best neighbor is better or random number is less than probability, update the solution
            if best_neighbor_cost < best_cost or random.random() < math.exp((best_cost - best_neighbor_cost) / temperature):
                best_solution = best_neighbor
                best_cost = best_neighbor_cost

                accepted_solutions.append((best_solution, best_cost))

        # Update temperature
        temperature *= cooling_rate
        # print(temperature)

    # Select the best solution among accepted ones
    best_solution, _ = min(accepted_solutions, key=lambda x: x[1])

    return best_solution




adjacency_list = build_adjacency_list()

num_vertices = int(input("Enter the number of vertices : "))

def set_inital_temperature(num_vertices):
    return num_vertices * 10
  
 
def set_inital_cooling_rate(num_edges): 
    return 1 / num_edges + 1




# Simulated Annealing parameters
initial_temperature =  set_inital_temperature(num_vertices)
cooling_rate = 0.99 #set_inital_cooling_rate(num_edges)
max_iterations = 50

# print(initial_temperature)
# print(cooling_rate)

# Call the simulated annealing function
# if adjacency_list.length() > 0:
best_solution = simulated_annealing(adjacency_list, initial_temperature, cooling_rate, max_iterations)
# else:
    

# Print the best solution
print("Best Solution:")
print(f"  Labels: {best_solution}")
print(f"  Cost: {cost_function(best_solution)}")


# Print all accepted solutions
for i, (solution, cost) in enumerate(accepted_solutions):
    print(f"Accepted Solution {i + 1}:")
    #print(f"  Labels: {solution}")
    print(f"  Cost: {cost}")
    print()


# Print the labels for each vertex
# for vertex, label in best_solution.items():
# print(f"Vertex {vertex} is assigned label {label}")





# adjacency_list = make_adjacency_list()
# vertex_labels = Heuristic_2(adjacency_list)
# print("Vertex Labels:", vertex_labels)
# print("Cost: ", sum(vertex_labels.values()))
# print(check_feasibility(vertex_labels, adjacency_list))