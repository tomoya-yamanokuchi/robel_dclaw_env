from xml.etree import ElementTree
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import join_with_mkdir
import xml.dom.minidom as md
from domain.xml_generation.element_tree.BodyTree import BodyTree
from domain.xml_generation.element_tree.JointTree import JointTree
from domain.xml_generation.element_tree.MujocoTree import MujocoTree



class PushingObjectElementTree:
    def __init__(self):
        self.mujoco_tree = MujocoTree()
        self.root = self.mujoco_tree.root


    def add_joint_tree(self):
        joint_tree = JointTree()
        self.root.insert(1, joint_tree.root)


    def add_body_tree(self, xy_pos: np.ndarray , mass: float):
        assert len(xy_pos.shape) == 2
        num_geom, dim = xy_pos.shape
        assert dim == 2
        body_tree = BodyTree()
        for id in range(num_geom):
            x_pos = str(xy_pos[id][0])
            y_pos = str(xy_pos[id][1])
            z_pos = str(0)
            pos   = " ".join((x_pos, y_pos, z_pos))
            body_tree.add_geometry(pos=pos, id=id, mass=mass)
        self.root.insert(1, body_tree.root)


    def save_xml(self):
        save_dir  = './domain/environment/model/pushing_object/'
        save_path = join_with_mkdir(save_dir, "object_current.xml", is_end_file=True)
        document  = md.parseString(ElementTree.tostring(self.root, 'utf-8'))
        with open(save_path, "w") as file:
            document.writexml(file, encoding='utf-8', newl='\n', indent='', addindent="\t")

if __name__ == '__main__':

    xy_position = np.random.randn(10, 2)*0.01
    mass        = 0.05

    # import ipdb; ipdb.set_trace()
    pusing_etree = PushingObjectElementTree()
    pusing_etree.add_joint_tree()
    pusing_etree.add_body_tree(xy_position, mass)
    pusing_etree.save_xml()
