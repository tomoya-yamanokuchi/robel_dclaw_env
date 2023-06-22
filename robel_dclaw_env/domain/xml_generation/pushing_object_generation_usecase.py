from ConvexHull2D import ConvexHull2D
from flattened_2d_meshgrid import flattened_2d_meshgrid
from PlotConvexHull import PlotConvexHull
from element_tree.PushingObjectXML import PushingObjectElementTree
from robel_dclaw_env.custom_service import normalize
import copy

from object_parameter.ObjectMass import ObjectMass
from object_parameter.ObjectFriction import ObjectFriction


convex        = ConvexHull2D(num_sample=7)
convex_origin = copy.deepcopy(convex)

# ---------- origin convex ---------
all_points           = flattened_2d_meshgrid(min=convex.min, max=convex.max, num_points_1axis=30)
inside_points_origin = convex_origin.get_inside_points(all_points)
plot_convex          = PlotConvexHull(convex_origin)
plot_convex.plot(all_points, inside_points_origin, "./convex_origin.png")

# ---------- aligned convex ---------
# inside_points = convex.get_inside_points(all_points)
convex.hull.points[:, 0] += (inside_points_origin[:, 0].mean())*(-1)
convex.hull.points[:, 1] += (inside_points_origin[:, 1].mean())*(-1)
aligned_inside_points = convex.get_inside_points(all_points)
plot_convex           = PlotConvexHull(convex)
plot_convex.plot(all_points, aligned_inside_points, "./convex_alinged.png")

# ---------- aligned convex ---------
nomalized_inside_points = normalize(aligned_inside_points, x_min=-1.0, x_max=1.0, m=-0.03, M=0.03)
num_inside_points       = nomalized_inside_points.shape[0]

object_mass     = ObjectMass()
object_friction = ObjectFriction()

# import ipdb; ipdb.set_trace()
pusing_etree = PushingObjectElementTree()
pusing_etree.add_joint_tree()
pusing_etree.add_body_tree(
    xy_pos   = nomalized_inside_points,
    mass     = object_mass.unit_inside_cylinder_mass(num_inside_points),
    friction = object_friction.unit_inside_cylinder_mass(num_inside_points),
)
pusing_etree.save_xml()
