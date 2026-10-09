from .preprocessing import RouteSetBuilder, RouteSpec, overlap_ratio
import heapq
import numpy as np


class OccupancyRouteSetBuilder:
    """A* over the current TartanGround coarse occupancy observation.

    Sparse columns are grid-x/grid-y/grid-z/class. Class 1 is flat; static (3)
    and vegetation (5) are blocked. Unknown remains traversable with a penalty.
    The fixed goal is supplied independently of the future path.
    """
    dependency_fields=("current_sparse_occupancy","fixed_goal_xy_local","resolution_m","route_spec")
    def __init__(self,spec=RouteSpec(),resolution_m=.5):self.spec,self.resolution_m=spec,resolution_m
    def build(self,sparse,fixed_goal_xy_local):
        x=np.asarray(sparse); n=int(max(250,x[:,:2].max()+1)); blocked=np.zeros((n,n),bool); flat=np.zeros((n,n),bool)
        blocked[x[np.isin(x[:,3],[3,5]),0],x[np.isin(x[:,3],[3,5]),1]]=True
        flat[x[x[:,3]==1,0],x[x[:,3]==1,1]]=True
        nominal_start=(n//2,n//2); delta=np.rint(np.asarray(fixed_goal_xy_local)/self.resolution_m).astype(int)
        raw_goal=(nominal_start[0]+int(delta[0]),nominal_start[1]+int(delta[1]));out_of_bounds=not(0<=raw_goal[0]<n and 0<=raw_goal[1]<n)
        if out_of_bounds:return self._empty(nominal_start,raw_goal,"goal_out_of_bounds",{"out_of_bounds":True})
        start_occ=bool(blocked[nominal_start]);goal_occ=bool(blocked[raw_goal]);start=self._nearest_free(blocked,nominal_start);goal=self._nearest_free(blocked,raw_goal)
        if start is None:return self._empty(nominal_start,raw_goal,"start_occupied_no_free_cell",{"start_occupied":start_occ})
        if goal is None:return self._empty(nominal_start,raw_goal,"goal_occupied_no_free_cell",{"goal_occupied":goal_occ})
        base=np.where(blocked,np.inf,np.where(flat,1.0,2.5))
        paths=[];cost=base.copy()
        for _ in range(self.spec.max_candidates):
            grid=self._astar(cost,start,goal)
            if not grid:break
            local=(np.asarray(grid,float)-np.asarray(start))*self.resolution_m
            dense=RouteSetBuilder(self.spec)._resample(local)
            if all(overlap_ratio(dense,q)<self.spec.dedup_overlap_threshold or overlap_ratio(q,dense)<self.spec.dedup_overlap_threshold for q in paths):paths.append(dense)
            for i,j in grid:cost[max(0,i-2):i+3,max(0,j-2):j+3]+=2.0
        arr=np.zeros((self.spec.max_candidates,self.spec.points_per_candidate,2),np.float32);mask=np.zeros(self.spec.max_candidates,bool)
        for i,p in enumerate(paths):arr[i]=p;mask[i]=True
        overlaps=[overlap_ratio(paths[i],paths[j]) for i in range(len(paths)) for j in range(i)]
        reason="ok" if paths else "astar_disconnected"
        return {"route_candidates_xy":arr,"route_candidate_mask":mask,"route_source":"map_goal","route_spec_hash":self.spec.hash,"map_source":"current_coarse_occupancy","grid_start":start,"grid_goal":goal,"diagnostics":{"reason":reason,"out_of_bounds":False,"start_occupied":start_occ,"goal_occupied":goal_occ,"start_adjusted":start!=nominal_start,"goal_adjusted":goal!=raw_goal,"unknown_fraction":float((~flat&~blocked).mean()),"candidate_count":len(paths),"max_pair_overlap":max(overlaps) if overlaps else 0.0,"dedup_overlap_threshold":self.spec.dedup_overlap_threshold}}
    def _nearest_free(self,blocked,point,radius=12):
        if not blocked[point]:return point
        for r in range(1,radius+1):
            for i in range(max(0,point[0]-r),min(blocked.shape[0],point[0]+r+1)):
                for j in (point[1]-r,point[1]+r):
                    if 0<=j<blocked.shape[1] and not blocked[i,j]:return (i,j)
            for j in range(max(0,point[1]-r),min(blocked.shape[1],point[1]+r+1)):
                for i in (point[0]-r,point[0]+r):
                    if 0<=i<blocked.shape[0] and not blocked[i,j]:return (i,j)
        return None
    def _empty(self,start,goal,reason,extra):
        arr=np.zeros((self.spec.max_candidates,self.spec.points_per_candidate,2),np.float32);mask=np.zeros(self.spec.max_candidates,bool)
        return {"route_candidates_xy":arr,"route_candidate_mask":mask,"route_source":"map_goal","route_spec_hash":self.spec.hash,"map_source":"current_coarse_occupancy","grid_start":start,"grid_goal":goal,"diagnostics":{"reason":reason,"candidate_count":0,**extra}}
    def _astar(self,cost,start,goal):
        q=[(0.0,start)];came={};g={start:0.0};seen=set()
        while q:
            _,u=heapq.heappop(q)
            if u in seen:continue
            seen.add(u)
            if u==goal:
                p=[u]
                while u in came:u=came[u];p.append(u)
                return p[::-1]
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                v=(u[0]+di,u[1]+dj)
                if not(0<=v[0]<cost.shape[0] and 0<=v[1]<cost.shape[1]) or not np.isfinite(cost[v]):continue
                z=g[u]+float(cost[v])
                if z<g.get(v,float("inf")):
                    g[v]=z;came[v]=u;h=abs(v[0]-goal[0])+abs(v[1]-goal[1]);heapq.heappush(q,(z+h,v))
        return []

__all__=["RouteSetBuilder","RouteSpec","overlap_ratio","OccupancyRouteSetBuilder"]
