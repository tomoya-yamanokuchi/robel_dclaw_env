from ConvexHull2D import ConvexHull2D
from flattened_2d_meshgrid import flattened_2d_meshgrid
from PlotConvexHull import PlotConvexHull
from element_tree.PushingObjectXML import PushingObjectElementTree
from custom_service import normalize
import copy

convex        = ConvexHull2D(num_sample=7)
convex_origin = copy.deepcopy(convex)

x_mean = convex.hull.points[:, 0].mean()
y_mean = convex.hull.points[:, 1].mean()
print("convex mean = ({}, {})".format(x_mean, y_mean))
convex.hull.points[:, 0] -= x_mean
convex.hull.points[:, 1] -= y_mean
x_mean = convex.hull.points[:, 0].mean()
y_mean = convex.hull.points[:, 1].mean()
print("convex mean = ({}, {})".format(x_mean, y_mean))

# import ipdb; ipdb.set_trace()
all_points    = flattened_2d_meshgrid(min=convex.min, max=convex.max, num_points_1axis=30)
inside_points = convex.get_inside_points(all_points)
plot_convex   = PlotConvexHull(convex)
plot_convex.plot(all_points, inside_points, "./convex_alinged.png")


inside_points_origin = convex_origin.get_inside_points(all_points)
plot_convex   = PlotConvexHull(convex_origin)
plot_convex.plot(all_points, inside_points_origin, "./convex_origin.png")


import ipdb; ipdb.set_trace()


inside_points = normalize(inside_points, x_min=-1.0, x_max=1.0, m=-0.03, M=0.03)

mass = (0.05 / inside_points.shape[0])

# import ipdb; ipdb.set_trace()
pusing_etree = PushingObjectElementTree()
pusing_etree.add_joint_tree()
pusing_etree.add_body_tree(inside_points, mass)
pusing_etree.save_xml()
