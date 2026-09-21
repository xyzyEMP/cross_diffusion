from __future__ import annotations
import heapq
import numpy as np
from scipy.ndimage import binary_dilation

def inflate(blocked,radius_cells):
    if radius_cells<=0:return np.asarray(blocked,bool).copy()
    extent=int(np.ceil(radius_cells));y,x=np.ogrid[-extent:extent+1,-extent:extent+1]
    return binary_dilation(np.asarray(blocked,bool),structure=x*x+y*y<=radius_cells*radius_cells)

def astar(blocked,start,goal,resolution_m=0.5):
    blocked=np.asarray(blocked,bool);start=tuple(start);goal=tuple(goal)
    if blocked[start] or blocked[goal]:return None,float("inf")
    moves=((1,0,1.),(-1,0,1.),(0,1,1.),(0,-1,1.),(1,1,2**.5),(1,-1,2**.5),(-1,1,2**.5),(-1,-1,2**.5))
    q=[(0.,start)];g={start:0.};came={};seen=set()
    while q:
        _,u=heapq.heappop(q)
        if u in seen:continue
        seen.add(u)
        if u==goal:
            p=[u]
            while u in came:u=came[u];p.append(u)
            p=p[::-1];return np.asarray(p,int),g[goal]*resolution_m
        for di,dj,c in moves:
            v=(u[0]+di,u[1]+dj)
            if not(0<=v[0]<blocked.shape[0] and 0<=v[1]<blocked.shape[1]) or blocked[v]:continue
            z=g[u]+c
            if z<g.get(v,float("inf")):
                g[v]=z;came[v]=u;h=np.hypot(v[0]-goal[0],v[1]-goal[1]);heapq.heappush(q,(z+h,v))
    return None,float("inf")

def sparse_grid(sparse):
    x=np.asarray(sparse);n=int(max(250,x[:,:2].max()+1));blocked=np.zeros((n,n),bool)
    q=x[np.isin(x[:,3],[3,5]),:2];blocked[q[:,0],q[:,1]]=True
    return blocked

def nearest_free(blocked, cell):
    """Return the closest free grid cell; deterministic row-major tie break."""
    cell=np.asarray(cell,int)
    if (0<=cell[0]<blocked.shape[0] and 0<=cell[1]<blocked.shape[1]
            and not blocked[tuple(cell)]): return tuple(cell)
    free=np.argwhere(~blocked)
    if not len(free): return None
    d=((free-cell)**2).sum(1)
    return tuple(free[np.argmin(d)])
