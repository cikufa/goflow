"""Extract actual calibration video keyframes, with their recorded phase labels."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=Path)
args=parser.parse_args()
with np.load(args.folder/'trajectories.npz') as data: phase=data['phase'][:,0]
reader=imageio.get_reader(str(args.folder/'handoff.mp4'))
frames=reader.count_frames()
end=lambda value:int(np.flatnonzero(phase==value)[-1])
items=[(0,'Initial scene'),(end(0)+1,'Visible-pose approach'),(end(1)+1,'Measured attachment / close'),
       (end(2),'Grasp closed'),(end(4),'Lift and hold'),(end(5),'Transport waypoint'),
       (end(7),'Staged INSERT initialization'),((end(7)+frames-1)//2,'Oracle INSERT'),(frames-1,'Final oracle state')]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
sheet=Image.new('RGB',(1440,900),'white');draw=ImageDraw.Draw(sheet)
for i,(index,label) in enumerate(items):
    image=Image.fromarray(reader.get_data(index));image.thumbnail((480,270))
    x=(i%3)*480;y=(i//3)*300;sheet.paste(image,(x,y))
    draw.text((x+8,y+272),label,font=font,fill='black')
sheet.save(args.folder/'contact_sheet.png')
reader.close()
print(args.folder/'contact_sheet.png')
