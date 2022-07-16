

def actionmapping(toward, vel):
    if toward == 119:
        return [0.1*vel, 0.1*vel]
    elif toward == 115:
        return [-0.1*vel, -0.1*vel]
    elif toward == 97:
        return [0.0, 0.1*vel]
    elif toward == 100:
        return [0.1*vel, 0.0]
    else:
        return [0, 0]