"""Explicit mixed-fleet demand, independent of tape length and query answers.

Counts describe allocation classes, not the number moving at any instant.
Long-stay cars never become failed trips disguised as static vehicles.
"""

import math
from dataclasses import dataclass

MODES = ("curb_trip", "curb_parked", "lot_parked")


@dataclass(frozen=True)
class VehicleRequest:
    index: int
    asset: str
    mode: str

    @property
    def actor_id(self):
        return f"vehicle-{self.index:03d}"

    @property
    def initial_purpose(self):
        return "parking_lot_long_stay" if self.mode == "lot_parked" else "resident_parking"


def requests(count, vehicle_assets, config):
    """Retain legacy IDs/assets; require explicit counts to sum exactly.

    Allocation order reserves moving routes before static parking. Appending
    parked cars cannot change existing trip IDs or their departure RNG streams.
    A failed admission is never moved into another allocation class.
    """
    if type(count) is not int or count < 0:
        raise ValueError("Vehicle count must be a nonnegative integer")
    if not vehicle_assets or any(not isinstance(x, str) or not x for x in vehicle_assets):
        raise ValueError("vehicle_assets must contain asset identifiers")
    population = config.get("population")
    if population is None:
        share = float(config.get("trip_share", 1.0))
        if not math.isfinite(share) or not 0 <= share <= 1:
            raise ValueError("trip_share must be between zero and one")
        trips = round(count * share)
        counts = dict(curb_trip=trips, curb_parked=count - trips, lot_parked=0)
    else:
        if "trip_share" in config:
            raise ValueError("Use either explicit population counts or trip_share, not both")
        if not isinstance(population, dict) or set(population) != set(MODES):
            raise ValueError("population must specify curb_trip, curb_parked and lot_parked")
        if any(type(n) is not int or n < 0 for n in population.values()):
            raise ValueError("Fleet population counts must be nonnegative integers")
        if sum(population.values()) != count:
            raise ValueError("Fleet population must sum to the requested vehicle count")
        counts = population
    modes = [mode for mode in MODES for _ in range(counts[mode])]
    return [VehicleRequest(i, vehicle_assets[i % len(vehicle_assets)], mode) for i, mode in enumerate(modes)]


def composition(roster):
    return {mode: sum(r.mode == mode for r in roster) for mode in MODES}
