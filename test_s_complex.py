import gudhi

points = [[0,0], [0,1], [1,1], [1.1,0]]

rips = gudhi.RipsComplex(points, max_edge_length=2) # max edge length is our max radius

simplex_tree = rips.create_simplex_tree(max_dimension=1)

result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(simplex_tree.num_vertices()) + ' vertices.'
print(result_str)
fmt = '%s -> %.2f'
for filtered_value in simplex_tree.get_filtration():
    print(fmt % tuple(filtered_value))
