import cv2, os


def save_mpc_planning_images(images: list, save_dir: str, fname: str):
    for step, img in enumerate(images):
        # if step == 0: continue
        fname_add = "_step{}.png".format(step)
        cv2.imwrite(
            filename = os.path.join(save_dir, fname + fname_add),
            img      = img,
        )
