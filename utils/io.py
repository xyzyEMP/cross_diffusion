from mmengine import fileio
import json

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict
