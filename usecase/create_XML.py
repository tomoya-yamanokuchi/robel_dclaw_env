import os
import numpy as np
from xml.etree.ElementTree import Element, ElementTree, SubElement, tostring
import xml.dom.minidom as md
from xml.sax import xmlreader


class GenerateClawTip:
    def __init__(self, claw_name="FFL12"):
        self.claw_name = claw_name
        self.root      = Element("mujocoinclude")
        finger         = self.add_finger()
        _              = self.add_optoforce(finger)


    def add_finger(self):
        # ------------ body ------------
        tree = SubElement(self.root,
            "body",
            attrib = {
                "name"       : "{}_tip".format(self.claw_name),
                "pos"        : "0 0 0",
                "childclass" : "dclaw3xh",
            }
        )
        # ------------------------------

        # visual
        SubElement(tree,
            "geom",
            attrib = {
                "name"    : "{}_plastic_finger".format(self.claw_name),
                "mesh"    : "finger_for_optoforce",
                "pos"     : "0 0 0",
                "euler"   : "0 0 1.57",
                "material": "{}_plastic_finger_mat".format(self.claw_name),
            }
        )

        # physical (底面)
        SubElement(tree,
            "geom",
            attrib = {
                "class": "phy_plastic",
                "type" : "box",
                "pos"  : "0 0 0.00254",
                "size" : "0.0241 0.014 0.00254",
                "mass" : ".007",
            }
        )

        # physical (指本体)
        SubElement(tree,
            "geom",
            attrib = {
                "class": "phy_plastic",
                "type" : "cylinder",
                "pos"  : "0 0 0.02332",
                "size" : "0.0115 0.01732",
                "mass" : ".009",
            }
        )
        return tree


    def add_optoforce(self, tree):
        # ------------ body ------------
        tree = SubElement(tree,
            "body",
            attrib = {
                "name"    : "{}_optoforce".format(self.claw_name),
                "pos"     : "0 0 0.073",
            }
        )
        # ------------------------------

        SubElement(tree,
            "geom",
            attrib = {
                "class": "phy_optoforce",
                "name" : "{}_phy_optoforce".format(self.claw_name),
            }
        )

        SubElement(tree,
            "site",
            attrib = {
                "class": "site_optoforce",
                "name" : "{}_site_optoforce".format(self.claw_name),
            }
        )




if __name__ == '__main__':
    compiler = GenerateClawTip()
    import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
    from domain.environment.XMLRepository import XMLRepository

    repository_path = os.path.expanduser("~") # robel-dclaw-envのリポジトリを置いているパス

    repository = XMLRepository()
    repository.writeXml(
        rootElement = compiler.root,
        save_path   = repository_path + "/robel-dclaw-env/domain/environment/model/robot/assets/abc.xml"
    )