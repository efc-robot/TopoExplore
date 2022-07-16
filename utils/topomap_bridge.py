from ros_topoexplore.msg import BinaryArrayMsg, EdgeMsg, VertexMsg, TopoMapMsg
from TopoMap import Vertex, Edge, TopologicalMap
import numpy as np
import sys

def TopomapToMessage(Topomap):
    topomap_message = TopoMapMsg()
    for i in range(len(Topomap.vertex)):
        vertexmsg = VertexMsg()
        vertexmsg.robot_name = Topomap.vertex[i].robot_name
        vertexmsg.id = Topomap.vertex[i].id
        vertexmsg.pose = Topomap.vertex[i].pose
        vertexmsg.descriptor = Topomap.vertex[i].descriptor.tolist()
        vertexmsg.navigableDirection = Topomap.vertex[i].navigableDirection
        for j in range(len(Topomap.vertex[i].frontierPoints)):
            frointpoint = BinaryArrayMsg()
            frointpoint.point = Topomap.vertex[i].frontierPoints[j].tolist()
            vertexmsg.frontierPoints.append(frointpoint)
        vertexmsg.frontierDistance = Topomap.vertex[i].frontierDistance
        topomap_message.vertex.append(vertexmsg)
    for i in range(len(Topomap.edge)):
        edgemsg = EdgeMsg()
        edgemsg.id = Topomap.edge[i].id
        edgemsg.robot_name1 = Topomap.edge[i].link[0][0]
        edgemsg.robot_name2 = Topomap.edge[i].link[1][0]
        edgemsg.id1 = Topomap.edge[i].link[0][1]
        edgemsg.id2 = Topomap.edge[i].link[1][1]
        topomap_message.edge.append(edgemsg)
    topomap_message.threshold = Topomap.threshold
    topomap_message.center = Topomap.center.tolist()
    topomap_message.offset_angle = Topomap.offset_angle

    return topomap_message

def MessageToTopomap(topomap_message):
    Topomap = TopologicalMap()
    for i in range(len(topomap_message.vertex)):
        vertex = Vertex()
        vertex.robot_name = topomap_message.vertex[i].robot_name
        vertex.id = topomap_message.vertex[i].id
        vertex.pose = topomap_message.vertex[i].pose
        vertex.descriptor = np.asarray(topomap_message.vertex[i].descriptor)
        vertex.navigableDirection = list(topomap_message.vertex[i].navigableDirection)
        for j in range(len(topomap_message.vertex[i].frontierPoints)):
            vertex.frontierPoints.append(np.asarray(topomap_message.vertex[i].frontierPoints[j].point))
        vertex.frontierDistance = list(topomap_message.vertex[i].frontierDistance)
        Topomap.vertex.append(vertex)
    for i in range(len(topomap_message.edge)):
        e = topomap_message.edge[i]
        link = [[e.robot_name1, e.id1], [e.robot_name2, e.id2]]
        edge = Edge(e.id, link)
        Topomap.edge.append(edge)
    Topomap.threshold = topomap_message.threshold
    Topomap.center = np.asarray(topomap_message.center)
    Topomap.offset_angle = topomap_message.offset_angle

    return Topomap
        