


class ObjectMass:
    def __init__(self, mass: float = 0.05):
        self.mass = mass


    def unit_inside_cylinder_mass(self, num_inside_cylinder):
        return (self.mass / num_inside_cylinder)
