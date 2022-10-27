
import math
import sys
from typing_extensions import TypedDict
import typing
from PIL import Image, ImageDraw
import numpy as np
from scipy import interpolate
from shapely.geometry import LineString
from marker_annotation_refine.alpha_shape import alpha_shape, edges_to_indices

Point = typing.Tuple[int, int]

# TODO: generate python types from annotation-api
MarkerLine = TypedDict(
  'MarkerLine',
  {
    'brushSize': int,
    'points': typing.List[Point],
    't': bool
  }
)

def flip_y(
  h,
  points : typing.List[Point]
):

  return [
    (p[0], h - p[1]) for p in points
  ]

def get_bounding_box(
  lines : typing.List[MarkerLine]
) -> typing.Tuple[int, int, int, int]:

  x0 : int = sys.maxsize
  y0 : int = sys.maxsize
  x1 : int = 0
  y1 : int = 0

  for line in lines:

    r = math.ceil(line['brushSize'] / 2)

    for p in line['points']:

      if p[0] - r < x0:

        x0 = p[0] - r

      if p[0] + r > x1:

        x1 = p[0] + r

      if p[1] - r < y0:

        y0 = p[1] - r

      if p[1] + r > y1:

        y1 = p[1] + r
 
 
  return (
    x0,
    y0, 
    x1 - x0,
    y1 - y0
  )

def draw_single_line(
  draw : ImageDraw.ImageDraw,
  x0 : int, 
  y0 : int,
  brush_size : int,
  points : typing.List[Point],
  intensity = 1
):

  r = brush_size / 2

  draw.ellipse(
    (
      points[0][0] - r - x0,
      points[0][1] - r - y0,
      points[0][0] + r - x0,
      points[0][1] + r - y0
    ),
    fill=intensity
  )

  draw.ellipse(
    (
      points[-1][0] - r - x0,
      points[-1][1] - r - y0,
      points[-1][0] + r - x0,
      points[-1][1] + r - y0
    ),
    fill=intensity
  )

  draw.line(
    [*[(p[0] - x0, p[1] - y0) for p in points], (points[0][0] - x0, points[0][1] - y0)], 
    width=brush_size, 
    fill=intensity,
    joint='curve'
  )

def draw_marker(
  lines : typing.List[MarkerLine],
  padding : int,
  highlight_center=True
):

  '''
  Draws one marker annotation as an image.
  Returns [origin, Image]
  '''

  x,y,w,h = get_bounding_box(lines)
  
  x -= int(padding // 2)
  y -= int(padding // 2)
  w += int(padding)
  h += int(padding)

  res = np.zeros((h, w))

  for line in lines:

    img = Image.new('F', (w, h))

    draw = ImageDraw.Draw(img)

    draw_single_line(
      draw, 
      x,
      y,
      line['brushSize'], 
      line['points']
    )

    if highlight_center:

      draw_single_line(
        draw, 
        x,
        y,
        1, 
        line['points'],
        2
      )

    # erase or draw the line
    if line['t']:

      res += np.array(img)

    else:

      res -= np.array(img) 

  return (x,y), res/np.max(res)


def grow_line(marker : MarkerLine):

  return LineString(marker['points']).buffer(marker['brushSize'] / 2)
