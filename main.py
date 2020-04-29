
from cma import purecma as pcma
import gmmCma as gcma
import time

import numpy as np
import vectormath as vmath
import math

import tkinter

# Make sure to have the server side running in CoppeliaSim: 
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

class CoppeliaJointAdjuster:

    jointCount = 7
    def __init__(self, clientID):
        self.clientID = clientID
        self.joints = []

        for i in range(1, CoppeliaJointAdjuster.jointCount + 1):
            jointName = "RobotArmJoint" + str(i) + "#"
            _, joint = sim.simxGetObjectHandle(clientID, jointName, sim.simx_opmode_blocking)
            self.joints.append(joint)

        _, self.distanceHandle = sim.simxGetDistanceHandle(clientID, "Distance", sim.simx_opmode_blocking)

    def adjustJoints(self, x):

        # pause to load all joints at the same time
        sim.simxPauseCommunication(self.clientID, True)
        for i in range(len(self.joints)):
            sim.simxSetJointPosition(self.clientID, self.joints[i], x[i] * 0.01745329252, sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, False)
        
        time.sleep(0.02) # give the simulation time to catch up

        # return the distance between tip and target
        _, d = sim.simxReadDistance(self.clientID, self.distanceHandle, sim.simx_opmode_blocking)

        return d

# Vector2 class controls a point in 2-D space
class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __str__(self):
        return "<" + str(self.x) + ", " + str(self.y) + ">"

    def div(self, scalar):
        return Vector2(self.x / scalar, self.y / scalar)
    
    def tuple(self):
        return (self.x, self.y)

    def mag(self):
        return (self.x * self.x + self.y * self.y) ** 0.5

    def rotate(self, radians):
        magnitude = self.mag()
        self.x = math.cos(radians) * magnitude
        self.y = math.sin(radians) * magnitude
        return self

    def angle(self):
        return math.atan2(self.y, self.x)

# SelfIdentifying is a class which records a unique 
# identifier string upon instantiation in self.id
class SelfIdentifying:
    instances = 0
    def __init__(self):
        self.id = str(self.__class__.__name__) + str(SelfIdentifying.instances)
        SelfIdentifying.instances += 1

# CanvasGrid represents a grid on a canvas
class CanvasGrid(SelfIdentifying):

    def __init__(self, canvas, gridSize):
        SelfIdentifying.__init__(self)
        self.canvas = canvas
        self.gridSize = gridSize

    def draw(self):
        
        self.canvas.delete(self.id) # clear what this used to be
        cd = Vector2(self.canvas.winfo_width(), self.canvas.winfo_height())

        # create vertical grid lines
        for i in range(math.floor(cd.x / self.gridSize)):
            x = i*self.gridSize
            coords = (x, 0, x, cd.y)
            self.canvas.create_line(coords, fill="#CFCFCF", tags=self.id)

        # create horizontal grid lines
        for i in range(math.floor(cd.y / self.gridSize)):
            y = i*self.gridSize
            coords = (cd.x, y, 0, y)
            self.canvas.create_line(coords, fill="#CFCFCF", tags=self.id)

# JointedArm represents a 2D robotic arm on a canvas
class JointedArm(SelfIdentifying):

    def __init__(self, canvas, lengths, angles, offset):
        """
        Instantiate `JointedArm` object instance using `lengths`.

        Parameters
        ----------
            `canvas`: `tkinter.Canvas`
                canvas where the arm will be drawn
            `lengths`: `list`
                of numbers (like ``[3, 2, 1.2]``), lengths of arm segments
            `angles`: `list`
                of numbers (like ``[3, 2, 1.2]``), angles on joints in radians
        """
        SelfIdentifying.__init__(self)

        self.canvas = canvas
        self.jointCount = len(lengths)
        self.setLengths(lengths)
        self.setAngles(angles)

        self.offset = offset

    # creates objects that are seen when this is drawn
    def draw(self):

        self._updatePoints()

        radius = 10
        c = self.canvas

        c.delete(self.id) # clear this off the canvas before re-drawing

        # draw lengths of arm
        for i in range(len(self.lengths)):

            (x1, y1) = (self.jointPositions[i] + self.offset).tuple()
            (x2, y2) = (self.jointPositions[i+1] + self.offset).tuple()
            coords = (x1, y1, x2, y2)
            bb1 = x1 - radius, y1 - radius, x1 + radius, y1 + radius

            # draw length of arm
            c.create_line(coords, width=10, fill="#3F0000", activefill="#009F00", tags=("arm", self.id))
            # draw joint
            c.create_oval(bb1, fill="#BFBF7F", activefill="#009F00", tags=("joint", self.id))

        # draw X where the point of the arm is
        (x1, y1) = (self.jointPositions[len(self.lengths)] + self.offset).tuple()
        bb1 = x1 - radius, y1 - radius, x1 + radius, y1 + radius
        bb2 = x1 + radius, y1 - radius, x1 - radius, y1 + radius
        c.create_line(bb1, width=3, fill="#3FBF7F", tags=("tip", self.id))
        c.create_line(bb2, width=3, fill="#3FBF7F", tags=("tip", self.id))
        c.update() # force the image to change

    # sets the lengths of arm
    def setLengths(self, lengths):
        assert self.jointCount == len(lengths) # Lengths of arm do not match the number of joints
        self.lengths = [x for x in lengths]

    # sets the angles of arm used
    def setAngles(self, angles):
        assert self.jointCount == len(angles) # Angles of arm do not match the number of joints
        self.angles = [x for x in angles]

    # Updates the global joint positions 
    def _updatePoints(self):

        lastAngle = 0
        self.jointPositions = [Vector2(0, 0)]
        for i in range(len(self.lengths)):
            angle = self.angles[i]
            length = self.lengths[i]
            lastAngle += angle

            span = Vector2(length, 0)
            span.rotate(lastAngle)
            self.jointPositions.append(self.jointPositions[i] + span)
        
    def testAngles(self, angles, target):

        self.setAngles(angles)
        self._updatePoints()
        
        diff = self.jointPositions[len(angles)] - target
        
        return diff.mag()

# GradientDescent is used to calculate the optimal values for a function
# The class holds the functions for this functionality,
# but has no use when instantiated
class GradientDescent:

    @staticmethod
    def fmin(func, params, change, iterationCount=50):

        params = [x for x in params] # copy elements over
        small = 1.0e-3

        lastEvaluation = func(params)
        #change /= 1.2

        for _ in range(iterationCount):
            evaluation = func(params)

            if (evaluation < lastEvaluation): # evaluation has improved
                change *= 1.2 # speed up
            else: # evaluation has not improved
                change *= 0.5 # slow down
            
            gradient = []
            magSquared = 0.0

            for i in range(len(params)):
                params[i] += small # bump up param
                diff = func(params) - evaluation
                slope = diff / small
                gradient.append(slope)
                params[i] -= small # bump down param

                magSquared += gradient[i] ** 2.0

            magnitude = magSquared ** 0.5
            if (abs(magnitude) < small**2):
                multiplier = 0.0
            else:
                multiplier = change / magnitude

            for i in range(len(gradient)):
                params[i] -= gradient[i] * multiplier
            
            lastEvaluation = evaluation
            
        return [params, True]

def handleSimulation(clientID):
    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    ja = CoppeliaJointAdjuster(clientID)

    angleDist = lambda x: ja.adjustJoints(x)
    solution = gcma.fmin(angleDist, [0] * ja.jointCount, math.pi)
    ja.adjustJoints(solution[0])

# Opens a window which contains a robot arm 
# and options for learning to a goal position 
def openCustomGui():
    gui = tkinter.Tk()
    canvas = tkinter.Canvas(gui, width=800, height=600)
    canvas.pack()
    canvas.update()

    grid = CanvasGrid(canvas, 40)
    grid.draw()

    armLengths = [(2+x)*20 for x in range(7)]
    initialAngles = [-1.0, 0.5, -1.2, -0.6, 0.5, -1.5, -0.4]
    arm = JointedArm(canvas, armLengths, initialAngles, Vector2(400, 400))

    # PeriodicPerformer counts the number of times a given function is called
    # every `period` calls, it calls a secondary function (which takes no params)
    class PeriodicCaller:
        def __init__(self, op1, op2, period):
            self.count = 0
            self.op1 = op1
            self.op2 = op2
            self.period = period
            
        def operate(self, params):
            result = self.op1(params)
            self.count += 1 # count the operations
            if self.count >= self.period: # every so often
                self.count -= self.period
                self.op2()

            return result

    target = Vector2(0, -200) # the target position for the arm head
    distFromTarget = lambda x: arm.testAngles(x, target)
    canvasTarget = arm.offset + target
    # draw target point
    canvas.create_oval(canvasTarget.x - 10, canvasTarget.y - 10, canvasTarget.x + 10, canvasTarget.y + 10, fill="#AAAAFF", width=4, outline="#0000AA")
    
    arm.draw()

    pc = PeriodicCaller(distFromTarget, arm.draw, 20)

    def doGradient():
        angles, _ = GradientDescent.fmin(pc.operate, initialAngles, 0.1)
        arm.setAngles(angles)
        arm.draw()

    def doPcma():
        angles, _ = pcma.fmin(pc.operate, initialAngles, 0.1)
        arm.setAngles(angles)
        arm.draw()

    def doIcma():
        angles, _ = gcma.fmin(pc.operate, initialAngles, 0.1)
        arm.setAngles(angles)
        arm.draw()

    def reset():
        arm.setAngles(initialAngles)
        arm.draw()
    
    resetButton = tkinter.Button(canvas, text="Reset", command=reset)
    button1 = tkinter.Button(canvas, text="Gradient Descent", command=doGradient)
    button2 = tkinter.Button(canvas, text="CMA-ES", command=doPcma)
    button3 = tkinter.Button(canvas, text="GMM-ES", command=doIcma)
    canvas.create_window(300, 550, window=button1)
    canvas.create_window(400, 550, window=button2)
    canvas.create_window(500, 550, window=button3)
    canvas.create_window(400, 480, window=resetButton)

    tkinter.mainloop()


if __name__ == "__main__":
    
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=-1#sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
    if clientID!=-1:
        print ('Connected to remote API server')

        handleSimulation(clientID)

        # Now close the connection to CoppeliaSim:
        sim.simxFinish(clientID)
    else:
        print ('Failed connecting to remote API server')
        openCustomGui()