import os
import numpy as np
from xml.etree.ElementTree import Element, ElementTree, SubElement, tostring
import xml.dom.minidom as md
from xml.sax import xmlreader


class GenerateOptoforce:
    def __init__(self, claw_name="FFL12"):
        self.claw_name = claw_name
        self.root      = Element("mujocoinclude")
        _              = self.add_optoforce()


    def add_optoforce(self):
        # ------------ body ------------
        tree = SubElement(self.root,
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
    import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
    from domain.environment.XMLRepository import XMLRepository

    claw_name = "FFL12"
    compiler  = GenerateOptoforce(claw_name)

    repository_path = os.path.expanduser("~") # robel-dclaw-envのリポジトリを置いているパス
    file_path       = repository_path + "/robel-dclaw-env/domain/environment/model/robot/assets/optoforce_{}.xml".format(claw_name)

    repository = XMLRepository()

    with open(file_path, 'w') as f:
        f.write('')

    repository.writeXml(
        rootElement = compiler.root,
        save_path   = file_path
    )


    '''
    2022/7/21現状
    ・mujocoのXMLファイルが改造しにくすぎる（一つひとつの数値が何を意味しているのか不明）
    ・しかもロボットモデル全体が１つモデルファイルに記述されていてバグを誘発しやすい
    ・改造しやすいように物理的なパーツとxmlファイルを1対1対応させたい
    ・そのためにxmlのパーサーを使ってpythonでxmlファイルを作れるようにしている途中
    ・まだ序盤
    '''