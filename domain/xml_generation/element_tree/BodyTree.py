import xml.etree.ElementTree as ET


class BodyTree:
    def __init__(self):
        self.root = ET.Element('body')
        self.root.set( "name", "pushing_object")
        self.root.set(  "pos", "0 0 .01001")
        self.root.set("euler", "0 0 0")

        # common parameter
        self.type = "cylinder"
        self.size = ".0015 .01"
        self.rgba = "1 0 0 1"
        # self.add_geometry()


    def add_geometry(self, pos, id, mass):
        self._add_visual_geometry(pos, id)
        self._add_physical_geometry(pos, mass)


    def _add_visual_geometry(self, pos: str, id: int):
        geom = self.__add_common(pos)
        geom.set("class", "pushing_object_viz")
        geom.set("name", "object_geom_vis_{}".format(id))
        geom.set("rgba", self.rgba)


    def _add_physical_geometry(self, pos: str, mass):
        geom = self.__add_common(pos)
        geom.set("class", "pushing_object_phy")
        geom.set("mass", str(mass))


    def __add_common(self, pos):
        geom = ET.SubElement(self.root, 'geom')
        geom.set("type", self.type)
        geom.set("size", self.size)
        geom.set("pos" , pos)
        return geom
