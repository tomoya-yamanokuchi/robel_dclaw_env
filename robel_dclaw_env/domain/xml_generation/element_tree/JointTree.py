import xml.etree.ElementTree as ET


class JointTree:
    def __init__(self):
        self.root = ET.Element('joint')
        self.root.set( "name", "object_jnt")
        self.root.set( "type", "free")
