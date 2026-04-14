"""
seestar-tracker — real-time tracking of fast-moving targets with the ZWO Seestar S50.

Modules:
    s50_client    — async TCP/JSON client for the S50 control protocol
    adsb_feed     — ADS-B aircraft position ingestion
    satellite     — TLE-based satellite/ISS position computation
    coord_utils   — geodetic → az/alt coordinate transforms
    track_engine  — closed-loop tracking controller
    cli           — command-line entry points
"""

__version__ = "0.1.0"
