from xml.etree.ElementTree import Element, tostring
import xml.dom.minidom as md


class XMLRepository:
    def writeXml(self, rootElement: Element, save_path: str, encode: str="utf-8"):
        assert isinstance(rootElement, Element)
        assert type(save_path) == str
        assert type(encode)    == str

        document = md.parseString(tostring(rootElement, encode))
        document.writexml(
            writer    = open(save_path, "w"),
            newl      = "\n",
            indent    = "",
            addindent = "\t",
            encoding  = encode
        )


if __name__ == '__main__':
    repository = XMLRepository()