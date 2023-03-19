from .OnscreenViewer import OnscreenViewer
from .OffscreenViewer import OffscreenViewer


class ViewerFactory:
    def create(self, is_Offscreen: bool):
        if is_Offscreen: return OffscreenViewer
        return OnscreenViewer
