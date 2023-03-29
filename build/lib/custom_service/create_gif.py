from PIL import Image


def create_gif(
        images  : list,
        fname   : str = './output.gif',
        duration: int = 500
    ):

    PIL_images = []
    for img in images:
        PIL_images.append(Image.fromarray(img[:, :, ::-1]))

    PIL_images[0].save(
        fname,
        save_all      = True,
        append_images = PIL_images[1:],
        optimize      = False,
        duration      = duration,
        loop          = 0
    )
