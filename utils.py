def dot(dt, x1, x2):
    xdot = (x1 - x2) / dt
    return xdot


def ddot(dt, x1, x2, x3):
    xddot = (dot(dt, x1, x2) - dot(dt, x2, x3)) / dt
    return xddot

