import xml.etree.ElementTree as ET


class MujocoTree:
    def __init__(self):
        self.root = ET.Element('mujocoinclude')
        self.root.set( "name", "pushing_object")
