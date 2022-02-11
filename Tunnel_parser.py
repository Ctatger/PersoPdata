import json
import numpy as np
from matplotlib import lines

def Tunnel_Parser(raw=False):
    Tunnels_coord=[]
    with open('idf_tunnels.geojson', 'r') as f:
        data = json.load(f)
        
    for f in data['features']:
        if 'nom' not in f['properties'].keys():
            f['properties']['nom']='No Name'
    if raw:
        return data

    for i,tunnels in enumerate(data['features']):#enumerate inutile, a enlever
        lines=tunnels['geometry']['coordinates']
        Tunnels_coord.append(lines[0]+lines[-1])
    return Tunnels_coord


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    if (len(iterable)%2 != 0):
        iterable.append(iterable[-1])
    a = iter(iterable)
    return zip(a, a)


def Tunnel_Parser_full(raw=False):
    Tunnels_coord=[]
    with open('idf_tunnels.geojson', 'r') as f:
        data = json.load(f)
        
    for f in data['features']:
        if 'nom' not in f['properties'].keys():
            f['properties']['nom']='No Name'
    if raw:
        return data

    for i,tunnels in enumerate(data['features']):#enumerate inutile, a enlever
        lines=tunnels['geometry']['coordinates']
        for a,b in pairwise(lines):
            Tunnels_coord.append(a + b)
    return Tunnels_coord


if __name__=="__main__":

    l = [1,2,3,4,5,6,7]


    for x, y in pairwise(l):
        print("iter")
        print(x)
        print(y)

    print(len(Tunnel_Parser()))
    print(Tunnel_Parser_full())
    print(len(Tunnel_Parser_full()))