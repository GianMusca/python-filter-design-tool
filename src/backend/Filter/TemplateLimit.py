class Dot:                          #A dot is defined by two points in space
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_dot(self):
        return [self.x, self.y]


class Limit:                        #A limit is the square on the Attenuation graph
    def __init__(self, p1: Dot, p2: Dot, p3: Dot, p4: Dot):
        self.p1 = p1        #HIGH LEFT
        self.p2 = p2        #HIGH RIGHT
        self.p3 = p3        #LOW LEFT
        self.p4 = p4        #LOW RIGHT

    def get_limit(self):
        return [self.p1, self.p2, self.p3, self.p4]
