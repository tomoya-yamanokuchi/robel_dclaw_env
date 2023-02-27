




def print_joint_positions(x):
    print("claw1=[{: .2f},{: .2f},{: .2f}], claw2=[{: .2f},{: .2f},{: .2f}], claw3=[{: .2f},{: .2f},{: .2f}]".format(
                     x[0],   x[1],   x[2],            x[3],   x[4],   x[5],            x[6],   x[7],   x[8]))

def print_ctrl(x):
    print("claw1=[{: .2f},{: .2f},{: .2f}], claw2=[{: .2f},{: .2f},{: .2f}], claw3=[{: .2f},{: .2f},{: .2f}]".format(
                     x[0],   x[1],   x[2],            x[3],   x[4],   x[5],            x[6],   x[7],   x[8]))

def print_task_space_positions(x):
    print("claw1=[{: .2f},{: .2f}], claw2=[{: .2f},{: .2f}], claw3=[{: .2f},{: .2f}]".format(
                     x[0],   x[1],            x[2],   x[3],            x[4],   x[5],))
