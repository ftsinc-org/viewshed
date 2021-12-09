#!/usr/bin/env python
import tornado.ioloop
import tornado.web
import tornado
from tornado.options import define, options
from helpers import TileSampler, CoordSystem
import json
from geojson import Feature, Point, MultiLineString
import geojson
from algo import find_visible_polygons
import plyvel
import math
from shapely.ops import transform
from shapely.geometry import mapping

define("port", default="8888", help="http port to listen on")
define("zoom", default=12, help="web mercator zoom level of dem data")
define(
    "tile_template",
    default="http://localhost:8888/api/v1/tiles/{z}/{x}/{y}.tiff",
    help="url template where web mercator dem tiles can be fetched",
)
define("leveldb", default="")

db = None


class TileHandler(tornado.web.RequestHandler):
    async def get(self, z, x, y):
        if not db:
            raise tornado.web.HTTPError(503)
        try:
            z = int(z)
            x = int(x)
            y = int(y)
        except Exception:
            raise tornado.web.HTTPError(400, "tiles coordinates must be integers")
        data = db.get(str.encode("/{}/{}/{}.tiff".format(z, x, y)))
        if not data:
            raise tornado.web.HTTPError(404, "tile not found")
        self.set_header("Content-type", "image/tiff")
        self.write(data)
        self.finish()


class ApiHandler(tornado.web.RequestHandler):
    def write_api_response(self, format, obj):
        format = format.lower()
        if format == "geojson":
            self.set_header("Content-Type", "application/vnd.geo+json")
            self.write(geojson.dumps(obj))
        elif format == "html":
            self.render(
                "../html/viewer.html", title="Viewshed API", geojson=geojson.dumps(obj)
            )

    def write_json(self, obj):
        self.set_header("Content-Type", "application/javascript")
        self.write(json.dumps(obj))

    def write_error(self, status_code, exc_info=None, **kwargs):
        errortext = "Internal error"
        if exc_info:
            errortext = getattr(exc_info[1], "log_message", errortext)
        self.write_json({"status": "error", "code": status_code, "reason": errortext})


class ElevationHandler(ApiHandler):
    async def get(self, format):
        lng = self.get_argument("lng")
        lat = self.get_argument("lat")
        try:
            lnglat = map(float, (lng, lat))
        except Exception:
            raise tornado.web.HTTPError(400)
        sampler = TileSampler(url_template=options.tile_template)
        pixel = CoordSystem.lnglat_to_pixel(lnglat)
        print(
            "Getting elevation at lng,lat:%s,%s %s,%s:" % (lng, lat, pixel[0], pixel[1])
        )
        value = await sampler.sample_pixel(pixel)
        lnglat = CoordSystem.pixel_to_lnglat(pixel)
        self.write_api_response(
            format,
            Feature(
                geometry=Point(lnglat),
                properties={
                    "elevation": float(value),
                    "uiMapCenter": lnglat,
                    "uiPopupContent": "{} meters".format(float(value)),
                },
            ),
        )


class TopOfHillHandler(ApiHandler):
    async def get(self, format):
        lng = self.get_argument("lng")
        lat = self.get_argument("lat")
        radius = self.get_argument("radius", 1000)
        try:
            lng, lat, radius = map(float, (lng, lat, radius))
        except Exception:
            raise tornado.web.HTTPError(400)
        radius = CoordSystem.pixel_per_meter((lng, lat)) * radius  # meters -> pixels
        print(
            "Getting top of hill at lng: {}, lat: {}, radius:{}".format(
                lng, lat, radius
            )
        )
        center = CoordSystem.lnglat_to_pixel((lng, lat))
        sampler = TileSampler(url_template=options.tile_template)

        # Iterate over all points in the square which surrounds the circle
        max_elv = None
        max_pos = None

        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        top_left = (center[0] - radius, center[1] - radius)
        for x in xrange(top_left[0], top_left[0] + radius * 2):
            for y in xrange(top_left[1], top_left[1] + radius * 2):
                # Is it in the circle?
                if math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius:
                    # Find the elevation
                    elevation = await sampler.sample_pixel((x, y))
                    if max_elv is None or elevation > max_elv:
                        max_elv = elevation
                        max_pos = (x, y)

        lnglat = CoordSystem.pixel_to_lnglat(max_pos)
        self.write_api_response(
            format,
            Feature(
                geometry=Point(lnglat),
                properties={
                    "elevation": float(max_elv),
                    "uiMapCenter": lnglat,
                    "uiPopupContent": "{} meters\n({},{})".format(
                        float(max_elv), lnglat[1], lnglat[0]
                    ),
                },
            ),
        )


class ShedHandler(ApiHandler):
    async def get(self, format):
        lng = self.get_argument("lng")
        lat = self.get_argument("lat")
        altitude = self.get_argument("altitude")
        radius = self.get_argument("radius", 1000)
        abs_altitude = self.get_argument("abs_altitude", False)
        try:
            lng, lat, altitude, radius = map(float, (lng, lat, altitude, radius))
        except Exception:
            raise tornado.web.HTTPError(400)
        radius = CoordSystem.pixel_per_meter((lng, lat)) * radius  # meters -> pixels
        print(
            "Getting viewshed at lng: {}, lat: {}, altitude: {}, radius:{}".format(
                lng, lat, altitude, radius
            )
        )
        center = CoordSystem.lnglat_to_pixel((lng, lat))
        sampler = TileSampler(url_template=options.tile_template)
        # add relative altitude offset
        if not abs_altitude:
            offset = await sampler.sample_pixel(center)
        else:
            offset = 0

        polygons = await find_visible_polygons(
            center, radius, altitude + offset, sampler.sample_pixel
        )
        transformed_polygons = transform(CoordSystem.pixels_to_lnglat, polygons)
        self.write_api_response(
            format,
            Feature(
                geometry=mapping(transformed_polygons),
                properties={
                    "calculationAltitude": altitude,
                    "calculationRadius": float(self.get_argument("radius", 1000)),
                    "calculationLat": lat,
                    "calculationLng": lng,
                    "uiMapCenter": (lng, lat),
                    "uiPopupContent": "Viewshed at {} meters above sea level".format(
                        altitude + offset
                    ),
                },
            ),
        )


application = tornado.web.Application(
    [
        (
            r"/bundle\.js()",
            tornado.web.StaticFileHandler,
            {"path": "../html/bundle.js"},
        ),
        (
            r"/bundle\.css()",
            tornado.web.StaticFileHandler,
            {"path": "../html/bundle.css"},
        ),
        (
            r"/viewshed()",
            tornado.web.StaticFileHandler,
            {"path": "../html/viewshed.html"},
        ),
        (r"/api/v1/elevation/(\w+)", ElevationHandler),
        (r"/api/v1/topofhill/(\w+)", TopOfHillHandler),
        (r"/api/v1/viewshed/(\w+)", ShedHandler),
        (r"/api/v1/tiles/(\d+)/(\d+)/(\d+)\.tiff", TileHandler),
        (r".*", tornado.web.RedirectHandler, {"url": "/viewshed"}),
    ]
)

if __name__ == "__main__":
    tornado.options.parse_command_line()
    if options.leveldb:
        db = plyvel.DB(options.leveldb, create_if_missing=False)
    application.listen(options.port)
    print("listening on port %s" % options.port)
    tornado.ioloop.IOLoop.current().start()
