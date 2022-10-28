



import math
from typing import Callable
import numpy as np
from marker_annotation_refine.edge_detection import edge_detect
from skimage.color import rgb2gray

from skimage.transform import resize

from marker_annotation_refine.marker_annotation import \
  MarkerLine, grow_line


def fit_polygon(
  marker : MarkerLine,
  bounds : tuple[int, int],
  # (x, y, phi, dist) => cost
  cost_fnc : Callable[[int, int, float, float, int, int], float]
):

  '''
  Attempts to fit a polygon to the image according to the marker.
  Returns a list of points for each point (root) on the marker line and their respective cost values.
  Points are to be interpreted in polar coordinates around their root.
  Returns amplitude, phase, cost
  '''

  assert len(bounds) == 2, 'objective_map should be 2D'

  brush_size = marker['brushSize']
  line = marker['points']

  # number of points scattered radially around each vertex on the line
  nr = 24
  # number of distances the costmap is evaluated at
  nd = 30

  angles = np.linspace(0, 2*np.pi, nr)
  radii = np.linspace(-0.3*brush_size/2, 1.5*brush_size/2, nd)

  # where lx[phi_idx, r_idx] is the x coord. of the point at orientation angles[phi_idx] at radius radii[r_idx]
  lx = np.array([radii*math.cos(phi) for phi in angles], dtype=int)
  ly = np.array([radii*math.sin(phi) for phi in angles], dtype=int)

  # resulting points around the line
  raddii_per_point = np.zeros((len(line), len(angles)))
  # the cost for each point
  cost_per_point = np.zeros(raddii_per_point.shape)

  for p_idx,p in enumerate(line):

    x,y = p

    plx = lx + x
    ply = ly + y

    # TODO: this can probably be done without the for loop

    for phi_idx in range(len(angles)):

      # radial line at angle phi
      rplx,rply = plx[phi_idx], ply[phi_idx]

      # collect the objective values for each pixel on the radial line
      obj = np.array(
        [
          cost_fnc(j,i, angles[phi_idx], radii[r_idx], y, x) \
          if (
            i >= 0 and j >= 0 and
            i < bounds[0] and 
            j < bounds[1]
          ) \
          # fall back to Inf if coordinates are out of bounds (important so that radii can be mapped to obj)
          else np.Infinity \
          for r_idx, (i,j) in enumerate(zip(rply, rplx))
        ]
      )
    

      best_cost_idx = np.argmin(obj).item()
      # obj shares indices with radii
      raddii_per_point[p_idx, phi_idx] = radii[best_cost_idx]

      cost_per_point[p_idx, phi_idx] = obj[best_cost_idx]
      

    # end for

  # end for

  return raddii_per_point, np.array([angles]*len(line)), cost_per_point

def clean_polygon_fit(
  marker : MarkerLine, 
  amp : np.ndarray,
  phi : np.ndarray,
  cost : np.ndarray
):

  # cartesian offsets for each point on the line
  ox = np.cos(phi) * amp
  oy = np.sin(phi) * amp
  
  lx,ly = np.array(marker['points']).transpose()
  
  x = np.concatenate([ox[i] + lx[i] for i in range(len(lx))])
  y = np.concatenate([oy[i] + ly[i] for i in range(len(ly))])

  coords = np.array((x,y)).transpose()

  #edges = alpha_shape(coords, (marker['brushSize'] / 2.0))

  # TODO: remove outliers
  return coords

# end def fit_polygon

if __name__ == '__main__':

  import os

  from dotenv import load_dotenv
  from matplotlib import pyplot as plt

  from marker_annotation_refine.marker_refine_dataset import PolygonDataset

  load_dotenv()

  ds = PolygonDataset(
    os.environ['CITYSCAPES_LOCATION'],
    'train',
  )

  for i,polygon in enumerate(ds):

    if i < 4: 
      continue

    x0,y0,_,_ = polygon.bounding_box()
    
    marker = polygon.random_marker()

    mx, my = np.array(marker['points']).transpose()
    
    mx, my = (mx - x0, my - y0)

    marker['points'] = np.transpose(
      (mx, my)
    ).tolist()

    marker_polygon = grow_line(marker)

    img = np.array(polygon.cropped_img())

    # gradient of the image
    dx,dy = np.gradient(rgb2gray(img))

    edges = edge_detect(img)

    scale = np.array(edges.shape) / img.shape[0:2]

    r = marker['brushSize'] / 2

    amp, phase, cost = fit_polygon(
      marker,
      img.shape[0:2],
      lambda j,i,phi,dist,rj,ri: \
        - edges[int(i*scale[0]),int(j*scale[1])] \
        + 1e-3*(dist - r)**2 \
        - ((rj-j)*dx[i,j] + (ri-i)*dy[i,j])
    )

    mx,my = np.array(
      clean_polygon_fit(
        marker,
        amp,
        phase,
        cost
      )
    ).transpose()

    cost = np.concatenate(cost)
    cost -= np.min(cost)
    cost /= np.max(cost)

    colors = [
      (c, 1, 1) for c in cost
    ]

    # mx,my = marker_polygon.exterior.coords.xy

    plt.subplot(1,2,1)

    plt.imshow(img)

    plt.scatter(mx, my, s=0.2, c=colors)

    plt.subplot(1,2,2)

    plt.imshow(resize(edges, img.shape[0:2]))

    plt.show()

