from .core import resample_fixed_arc

class RepresentationBridge:
    source_space="source_temporal_8s_80"
    research_space="fixed_arc_length_80"
    def to_research(self,se2,length_m): return resample_fixed_arc(se2,length_m,80)
